# 🧠 AI Segmentation Arena: A Comparative Analysis with LLM Insights

Welcome to **AI Segmentation Arena** — a futuristic playground where *pixels meet intelligence*. This interactive app lets you **compare** two of the most advanced computer vision models — **YOLOv8** and **SAM (Segment Anything Model)** — on any image you upload. Then, we go a step further by using an **LLM (Google Gemini)** to explain and evaluate the results intelligently.

---

## 🚀 Project Highlights

- ⚔️ **Model Face-off**: Compare YOLOv8's instance segmentation vs. SAM's universal masks — visually and quantitatively.
- 🤖 **LLM Integration**: Get detailed, natural-language insights from a large language model on performance, strengths, and best use cases.
- 🧪 **Real-Time Analysis**: See inference times, object counts, and smart summaries — all in a clean web UI powered by **Gradio**.
- 🧰 **Built for Developers and Researchers**: Modular structure, Docker-ready, Colab demo included.

---

## 📁 Folder Structure

```
weekend_project_01/
├── segmentation_backend/
│   ├── app.py                # Main Gradio app with ML and LLM logic
│   ├── dependencies.sh       # Bash script to install Python dependencies
│   ├── Dockerfile            # Docker setup for cloud deployment
│   ├── models/
│   │   └── sam_vit_h_4b8939.pth  # SAM model checkpoint
│   └── README.md             # This file!
```

---

## 🌐 Try it in Google Colab

Quickly test the core computer vision functionality (YOLOv8 + SAM, no LLM):

👉 [Run in Colab](https://colab.research.google.com/drive/1vKPHQlxpXJ9yKHNikfUPSfwVz7A6Wj7l?usp=sharing)

---

## 🛠️ Local Setup Guide (VS Code + WSL Recommended)

### 🧾 Prerequisites
- Python 3.10+
- Git
- VS Code
- [WSL2 + Ubuntu](https://learn.microsoft.com/en-us/windows/wsl/install) (for Windows users)

💡 *Windows users, enhance WSL performance with memory config:*

```ini
# C:\Users\<YourUsername>\.wslconfig
[wsl2]
memory=8GB
processors=4
swap=2GB
```

Then restart WSL:

```bash
wsl --shutdown
```

---

### 🔧 Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/AI-Segmentation-Arena.git
cd AI-Segmentation-Arena/segmentation_backend
```

2. **Set Up Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```

3. **Install Dependencies via Bash Script**
```bash
bash dependencies.sh
```

4. **Download SAM Model Checkpoint**
- Visit: [SAM GitHub Releases](https://github.com/facebookresearch/segment-anything)
- Download:
  - `sam_vit_h_4b8939.pth` (more accurate, slower) OR
  - `sam_vit_b_01ec64.pth` (lighter, faster)
- Place it into: `segmentation_backend/models/`

Then update `app.py`:
```python
SAM_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
```

5. **Set Your Google Gemini API Key**
Sign up at [Google AI Studio](https://makersuite.google.com/), generate an API key, then add to `app.py`:

```python
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```

---

## ▶️ Run the App Locally

```bash
python app.py
```

Visit: [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## 🧠 How It Works

1. Upload an image.
2. YOLOv8 segments objects. SAM segments everything.
3. LLM compares and comments on both results.
4. You analyze and download the insights.

---

## 📊 Model Insights

| Feature         | YOLOv8                        | SAM                          |
|----------------|-------------------------------|------------------------------|
| Inference Speed| ⚡ Fast                         | 🐢 Slower                    |
| Output         | Object detection & masks       | Universal masks              |
| LLM Feedback   | Explains pros, cons, use cases | Side-by-side comparison      |

---

## 🖼️ Sample Results

> You can upload screenshots or result images to the `results/` folder and link them here. For example:

![YOLOv8 vs SAM Comparison](results/sample_comparison.png)
![LLM Feedback](results/sample_llm_feedback.png)

---

## 🐳 Optional: Deploy to Google Cloud Run with Docker

If you want to host the app publicly, follow these Docker + GCP steps.

### Build Docker Image

```bash
docker build -t segmentation-arena .
```

### Push to Google Cloud (after setting up gcloud CLI)

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/segmentation-arena .
```

### Deploy to Cloud Run

```bash
gcloud run deploy segmentation-arena \
  --image gcr.io/YOUR_PROJECT_ID/segmentation-arena \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300s
```

---

### ☁️ Installing `gcloud` CLI in WSL (if needed)

If you get `Command 'gcloud' not found`, run:

```bash
sudo snap install google-cloud-cli --classic
```

Then initialize:

```bash
gcloud init
```

Follow the prompts to log in, select your project, and configure defaults.

---

## 🧠 Credits

- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- SAM: [Meta Research](https://github.com/facebookresearch/segment-anything)
- Gemini API: [Google AI Studio](https://ai.google.dev/)
- Web UI: [Gradio](https://gradio.app/)

---

## 🌟 Contribute or Star!

If this project helped you, consider ⭐ starring it or contributing!
