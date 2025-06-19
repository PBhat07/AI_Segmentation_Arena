import os
import cv2
import numpy as np
import time
import torch
import json
import httpx # For making async HTTP requests to the Gemini API
import gradio as gr # Import Gradio

# Try to import ML libraries; if not found, provide a clear message.
try:
    from ultralytics import YOLO
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import supervision as sv
except ImportError as e:
    raise RuntimeError(f"Missing required ML libraries: {e}. Please ensure 'ultralytics', 'segment-anything', and 'supervision' are installed.")

# --- Google Gemini API Configuration ---
# IMPORTANT: Replace "YOUR_GEMINI_API_KEY_HERE" with your actual Gemini API Key
GEMINI_API_KEY = "AIzaSyAO9F2SjynjLduLJ11o2vCK2RImQpf9jDY" # <-- PASTE YOUR GEMINI API KEY HERE
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


# --- Model Loading (Global variables to load models once when the script starts) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Server starting: Using device for models: {DEVICE}")

YOLO_MODEL_NAME = "yolov8n-seg.pt" # Nano segmentation model
SAM_MODEL_TYPE = "vit_b"           # SAM model type (vit_h, vit_l, vit_b)
SAM_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth" # Exact filename you downloaded for vit_b
SAM_CHECKPOINT_PATH = os.path.join("models", SAM_CHECKPOINT_FILENAME) # Path relative to app.py

yolo_model = None
sam_automask_generator = None
mask_annotator_global = None
box_annotator_global = None
label_annotator_global = None

# Load models and annotators globally (once)
try:
    yolo_model = YOLO(YOLO_MODEL_NAME)
    yolo_model.to(DEVICE)
    print(f"Loaded YOLOv8 model: {YOLO_MODEL_NAME}")

    # Ensure SAM checkpoint file exists before attempting to load
    if not os.path.exists(SAM_CHECKPOINT_PATH):
        raise FileNotFoundError(f"SAM checkpoint not found at: {SAM_CHECKPOINT_PATH}")

    sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    sam_model.to(device=DEVICE)
    sam_automask_generator = SamAutomaticMaskGenerator(sam_model)
    print(f"Loaded SAM model: {SAM_MODEL_TYPE} from {SAM_CHECKPOINT_PATH}")

    # Initializing supervisions annotators globally for efficiency
    mask_annotator_global = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    box_annotator_global = sv.BoxAnnotator()
    label_annotator_global = sv.LabelAnnotator()

    print("All models and annotators loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the SAM checkpoint file is present.")
    exit(1) # Exit if critical model is not found
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    exit(1) # Exit if other critical errors occur during loading


async def get_llm_insights(original_image_description, yolo_summary, sam_summary, metrics):
    """Calls the Gemini LLM to generate insights."""
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        return "Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual Gemini API Key to enable AI insights."

    prompt = f"""
    You are an AI assistant specialized in image segmentation analysis.
    Analyze the following segmentation results from YOLOv8 and SAM.
    The original image contains: {original_image_description}.

    **YOLOv8 Segmentation Summary (Model: {metrics['yolo_model_name']}):**
    {yolo_summary}
    Detected {metrics['yolo_objects_found']} objects.

    **SAM Segmentation Summary (Model: {metrics['sam_model_type']}):**
    {sam_summary}
    Generated {metrics['sam_masks_generated']} masks.

    **Performance Metrics:**
    YOLOv8 Inference Time: {metrics['yolo_inference_time_ms']:.2f} ms
    SAM Inference Time: {metrics['sam_inference_time_ms']:.2f} ms
    Image Resolution: {metrics['image_width']}x{metrics['image_height']} pixels

    **Task:**
    1.  Describe the key visual differences in how YOLOv8 and SAM segmented the image.
    2.  Discuss their respective strengths and weaknesses based on these results.
    3.  Analyze the performance differences in terms of inference time and the nature of their output (specific objects vs. everything).
    4.  Provide insights on the types of real-world applications each model might be best suited for, considering their inherent segmentation approach.
    5.  Keep the analysis concise and informative, in Markdown format.
    """

    chat_history = [{ "role": "user", "parts": [{ "text": prompt }] }]
    payload = { "contents": chat_history }

    headers = {'Content-Type': 'application/json'}
    params = {'key': GEMINI_API_KEY}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_API_URL, params=params, json=payload, timeout=90) # Increased timeout
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            print(f"LLM response structure unexpected: {result}")
            return "Could not generate insights from the AI."
    except httpx.RequestError as e:
        print(f"HTTP request to Gemini API failed: {e}")
        return f"Error connecting to AI: {e}. Please check your API key and network."
    except httpx.HTTPStatusError as e:
        print(f"Gemini API returned an error: {e.response.status_code} - {e.response.text}")
        return f"AI API error: {e.response.status_code}. Details: {e.response.text}"
    except Exception as e:
        print(f"An unexpected error occurred during LLM call: {e}")
        return f"An unexpected AI error occurred: {e}"


# --- Main processing function for Gradio ---
async def segment_and_analyze_image(image_rgb_input: np.ndarray):
    if yolo_model is None or sam_automask_generator is None:
        return (None, None, "Models are not loaded yet. Please try again or restart the server.", "Error: Models not loaded.")

    H, W, _ = image_rgb_input.shape

    results_metrics = {
        "yolo_inference_time_ms": 0,
        "sam_inference_time_ms": 0,
        "yolo_objects_found": 0,
        "sam_masks_generated": 0,
        "yolo_model_name": YOLO_MODEL_NAME,
        "sam_model_type": SAM_MODEL_TYPE,
        "image_width": W,
        "image_height": H
    }

    # --- Run YOLOv8 Segmentation ---
    start_time_yolo = time.perf_counter()
    yolo_raw_results = yolo_model(image_rgb_input, verbose=False)[0]
    end_time_yolo = time.perf_counter()
    results_metrics["yolo_inference_time_ms"] = (end_time_yolo - start_time_yolo) * 1000

    yolo_detections = sv.Detections.from_ultralytics(yolo_raw_results)
    results_metrics["yolo_objects_found"] = len(yolo_detections)

    annotated_yolo_image = image_rgb_input.copy()
    annotated_yolo_image = mask_annotator_global.annotate(
        scene=annotated_yolo_image,
        detections=yolo_detections
    )
    yolo_class_names = yolo_model.names
    yolo_labels = []
    yolo_summary_lines = []
    if yolo_detections.class_id is not None and yolo_detections.confidence is not None:
        for class_id, confidence in zip(yolo_detections.class_id, yolo_detections.confidence):
            class_name = yolo_class_names[class_id] if class_id < len(yolo_class_names) else f"Class {class_id}"
            yolo_labels.append(f"{class_name} {confidence:.2f}")
            yolo_summary_lines.append(f"- Detected a '{class_name}' with confidence {confidence:.2f}.")

    annotated_yolo_image = box_annotator_global.annotate(
        scene=annotated_yolo_image,
        detections=yolo_detections
    )
    annotated_yolo_image = label_annotator_global.annotate(
        scene=annotated_yolo_image,
        detections=yolo_detections,
        labels=yolo_labels
    )
    yolo_summary = "No objects detected by YOLOv8." if not yolo_summary_lines else "\n".join(yolo_summary_lines)


    # --- Run SAM Automatic Mask Generation ---
    start_time_sam = time.perf_counter()
    sam_auto_masks_raw = sam_automask_generator.generate(image_rgb_input)
    end_time_sam = time.perf_counter()
    results_metrics["sam_inference_time_ms"] = (end_time_sam - start_time_sam) * 1000
    results_metrics["sam_masks_generated"] = len(sam_auto_masks_raw)

    sam_detections_auto = sv.Detections.from_sam(sam_result=sam_auto_masks_raw)
    annotated_sam_image = image_rgb_input.copy()
    # FIX: Corrected typo from `deteated_sam_image` to `sam_detections_auto`
    annotated_sam_image = mask_annotator_global.annotate(
        scene=annotated_sam_image,
        detections=sam_detections_auto
    )
    sam_summary = f"SAM generated a total of {len(sam_auto_masks_raw)} masks, segmenting various distinct regions."


    # --- Original Image Description for LLM (Placeholder) ---
    # For a more advanced app, you could integrate an image captioning model here
    original_image_description = "an uploaded image"


    # --- Get LLM Insights ---
    llm_insights_text = await get_llm_insights(
        original_image_description,
        yolo_summary,
        sam_summary,
        results_metrics
    )

    # --- Format Metrics for display ---
    metrics_display = f"""
    ### Performance Metrics
    **YOLOv8 ({results_metrics['yolo_model_name']}):**
    - **Inference Time:** {results_metrics['yolo_inference_time_ms']:.2f} ms
    - **Objects Found:** {results_metrics['yolo_objects_found']}

    **SAM ({results_metrics['sam_model_type']}):**
    - **Inference Time:** {results_metrics['sam_inference_time_ms']:.2f} ms
    - **Masks Generated:** {results_metrics['sam_masks_generated']}

    **Image Resolution:** {results_metrics['image_width']}x{results_metrics['image_height']} pixels
    """

    # Gradio will display numpy arrays as images.
    # It will display strings as text or markdown.
    return (annotated_yolo_image, annotated_sam_image, metrics_display, llm_insights_text)


# --- Gradio Interface Setup ---
if __name__ == "__main__":
    # Define Gradio Interface components
    inputs = gr.Image(type="numpy", label="Upload Image", elem_id="upload-image-button")

    outputs = [
        gr.Image(type="numpy", label="YOLOv8 Segmentation", elem_id="yolo-image"),
        gr.Image(type="numpy", label="SAM Segmentation", elem_id="sam-image"),
        gr.Markdown(label="Performance Metrics", elem_id="metrics-output"),
        gr.Markdown(label="AI Insights & Comparison", elem_id="insights-output")
    ]

# Create the Gradio Interface
    demo = gr.Interface(
        fn=segment_and_analyze_image,
        inputs=inputs,
        outputs=outputs,
        title="AI Segmentation Arena",
        description="Upload an image to see side-by-side segmentation results from YOLOv8 and SAM, along with performance metrics and AI-generated insights.",
        allow_flagging="never", # Disable flagging
        live=False,
        # Applying the custom theme
        theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="pink",
    neutral_hue="zinc",
    font=[ "Inter", "sans-serif" ]
),
        css="""
        body {
            background-color: #0f0f0f !important;
            color: #ffffff !important;
            font-family: 'Inter', sans-serif !important;
        }

        #component-upload-image-button {
            border-radius: 9999px !important;
            background-image: linear-gradient(to right, var(--color-purple-600), var(--color-blue-500)) !important;
            transition: all 0.3s ease-in-out !important;
            transform: scale(1) !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
            padding: 0.25rem !important;
        }

        #component-upload-image-button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
        }

        #component-upload-image-button > button {
            background-color: #1a1a1a !important;
            border-radius: 9999px !important;
            color: #ffffff !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-weight: 600 !important;
            padding: 0.625rem 1.25rem !important;
            transition: all 0.075s ease-in !important;
        }

        #component-upload-image-button > button:hover {
            background-color: transparent !important;
        }

        /* Adjust image containers */
        #yolo-image, #sam-image {
            background-color: #1a1a1a !important;
            border-radius: 1rem !important;
            padding: 1.5rem !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
            border: 1px solid #374151 !important;
            transition: all 0.3s ease-in-out !important;
        }
        #yolo-image:hover { border-color: #EC4899 !important; }
        #sam-image:hover { border-color: #3B82F6 !important; }

        /* General block styling */
        .gradio-container {
            background-color: #0f0f0f !important;
            border-radius: 1.5rem !important;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25) !important;
            border: 1px solid #374151 !important;
            padding: 2.5rem !important;
        }

        .block-title, .label {
            color: ##ffffff !important;
            font-weight: 700 !important;
        }
        .label-wrap > label {
            font-weight: 600 !important;
        }

        /* Headings and text styling */
        h1.gr-text-center {
            font-size: 3.75rem !important;
            background-image: linear-gradient(to right, #A78BFA, #EC4899, #EF4444) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: ##ffffff !important;
            background-clip: text !important;
            color: ##ffffff !important;
            font-weight: 800 !important;
        }
        .gradio-h2 {
            font-size: 2rem !important;
            font-weight: 700 !important;
            text-align: center !important;
            margin-bottom: 2rem !important;
        }

        /* Markdown output */
        #metrics-output > h3, #insights-output > h3 {
            color: #A78BFA !important;
            font-weight: 700 !important;
            margin-top: 1em !important;
            margin-bottom: 0.5em !important;
        }
        #metrics-output > p, #insights-output > p {
            color: #f3f4f6 !important;
            margin-bottom: 0.5em !important;
            line-height: 1.6 !important;
        }
        #metrics-output strong, #insights-output strong {
            color: #FDBA74 !important;
            font-weight: 600 !important;
        }
        #metrics-output ul, #insights-output ul {
            list-style-type: disc !important;
            margin-left: 1.5em !important;
            margin-bottom: 1em !important;
            padding-left: 0 !important;
        }
        #metrics-output li, #insights-output li {
            margin-bottom: 0.25em !important;
        }

        /* Code blocks */
        .gr-markdown code {
            background-color: #333 !important;
            padding: 0.2em 0.4em !important;
            border-radius: 4px !important;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;
            color: #4CAF50 !important;
        }
        .gr-markdown pre {
            background-color: #222 !important;
            padding: 1em !important;
            border-radius: 8px !important;
            overflow-x: auto !important;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;
        }

        /* Gradient titles */
        .metrics-title {
            background-image: linear-gradient(to right, #FFD700, #FF69B4, #FF4500) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            color: #FFD700 !important;  /* Fallback color for non-webkit or low contrast */
            font-weight: 800 !important;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.6);  /* Optional: adds depth */
        }
        .insights-title {
            background-image: linear-gradient(to right, #4CAF50, #8BC34A) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            color: #4CAF50 !important;  /* Fallback color */
            font-weight: 800 !important;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.5);
        }

        /* Image placeholders */
        .gr-image-placeholder {
            background-color: #222 !important;
            border: 1px dashed #555 !important;
            color: #888 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            min-height: 200px !important;
            border-radius: 8px !important;
        }
        .gr-image-placeholder::before {
            content: "Image will appear here" !important;
            font-size: 1.125rem !important;
        }
        """
    )


    demo.launch(share=False)   # Removed server_name and server_port for default Gradio behavior