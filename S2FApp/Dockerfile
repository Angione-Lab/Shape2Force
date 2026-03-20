# Shape2Force (S2F) - Hugging Face Spaces
FROM python:3.10-slim

# Create user for HF Spaces (runs as UID 1000)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-docker.txt ./

# Install Python dependencies - CPU-only PyTorch to fit Space memory limits (avoids OOM)
# (psutil removed from requirements-docker.txt — optional CPU/RAM widget not used in app)
# PyTorch 2.2 + torchvision 0.17 (CPU) - match requirements.txt torch>=2.0
# Install numpy first with compatible version (<2) so torch.from_numpy works
RUN pip install --no-cache-dir "numpy>=1.20.0,<2.0" && \
    pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy app code (chown for HF Spaces permissions)
COPY --chown=user:user app.py predictor.py download_ckp.py ./
COPY --chown=user:user .streamlit/ .streamlit/
COPY --chown=user:user static/ static/
COPY --chown=user:user models/ models/
COPY --chown=user:user ui/ ui/
COPY --chown=user:user utils/ utils/
COPY --chown=user:user config/ config/
COPY --chown=user:user samples/ samples/
RUN mkdir -p ckp && chown user:user ckp

# Download checkpoints from Hugging Face if ckp is empty (for Space deployment)
# Requires HF_TOKEN secret in Space Settings (https://huggingface.co/settings/tokens)
# for private model repos. Set HF_MODEL_REPO to your model repo, e.g. Angione-Lab/Shape2Force
ARG HF_MODEL_REPO=Angione-Lab/Shape2Force
ENV HF_MODEL_REPO=${HF_MODEL_REPO}
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) python download_ckp.py

# Ensure ckp contents are readable by user
RUN chown -R user:user ckp

USER user

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
