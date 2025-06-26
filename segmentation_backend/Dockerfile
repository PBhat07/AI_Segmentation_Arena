# Use a Python base image
FROM python:3.10-slim-buster

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \  # <--- ADD THIS LINE
    && rm -rf /var/lib/apt/lists/*  
    # Clean up apt cache to keep image small


# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
# Use --no-cache-dir to save space. Adjust torch installation if you want specific CUDA versions.
# Ensure pip is up-to-date before installing other packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code and the models folder
COPY . .

# Expose the port Gradio runs on
EXPOSE 7860

# Command to run the Gradio application
CMD ["python", "app.py"]