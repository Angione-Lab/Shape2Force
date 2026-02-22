# Shape2Force (S2F) - Hugging Face Spaces
FROM python:3.10-slim

# Create user for HF Spaces (runs as UID 1000)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (exclude heavy training deps for smaller image)
RUN pip install --no-cache-dir \
    torch torchvision \
    numpy opencv-python streamlit matplotlib Pillow plotly \
    huggingface_hub

# Copy app code (chown for HF Spaces permissions)
COPY --chown=user:user app.py predictor.py download_ckp.py ./
COPY --chown=user:user models/ models/
COPY --chown=user:user utils/ utils/
COPY --chown=user:user config/ config/
COPY --chown=user:user sample/ sample/
RUN mkdir -p ckp && chown user:user ckp

# Download checkpoints from Hugging Face if ckp is empty (for Space deployment)
# Set HF_MODEL_REPO env to your model repo, e.g. kaveh/Shape2Force
ARG HF_MODEL_REPO=kaveh/Shape2Force
ENV HF_MODEL_REPO=${HF_MODEL_REPO}
RUN python download_ckp.py

# Ensure ckp contents are readable by user
RUN chown -R user:user ckp

USER user

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
