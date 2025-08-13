# Use official Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Create directories to store flair cached embeddings and model files
RUN mkdir -p \
    flair_cache_root \
	models

# Copy model files
COPY models/ /app/models/

# Copy flair cached embeddings files
COPY flair_cache_root/ /app/flair_cache_root/

# Copy necessary files - 1
COPY requirements.txt ./

# Install dependencies
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && rm -rf /root/.cache/pip
RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip
RUN pip install sentencepiece==0.1.95 && rm -rf /root/.cache/pip
RUN pip install --upgrade protobuf==3.20.* && rm -rf /root/.cache/pip
RUN pip install numpy==1.24.4 pandas==1.5.3 && rm -rf /root/.cache/pip

# Copy necessary files - 2
COPY app.py model_loader.py preload_model.py streamlit_app.py favicon.ico logo.png ./

# Expose port 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000 8501

# Preload model before starting API
CMD ["sh", "-c", "python preload_model.py && uvicorn app:app --host 0.0.0.0 --port 8000"]