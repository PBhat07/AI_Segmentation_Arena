

echo "🔧 Installing base dependencies..."
pip install --no-cache-dir gradio numpy httpx supervision

echo "🔧 Installing vision libraries..."
pip install --no-cache-dir opencv-python
pip install --no-cache-dir ultralytics
pip install --no-cache-dir segment-anything

echo "🔧 Installing PyTorch stack..."
pip install --no-cache-dir torch torchvision torchaudio

echo "✅ All installations complete."
