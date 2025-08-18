FROM python:3.9

WORKDIR /app

RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && rm -rf /root/.cache/pip

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip
RUN pip install sentencepiece==0.1.95 && rm -rf /root/.cache/pip
RUN pip install --upgrade protobuf==3.20.* && rm -rf /root/.cache/pip
RUN pip install numpy==1.24.4 pandas==1.5.3 && rm -rf /root/.cache/pip

RUN mkdir -p \
    flair_cache_root \
	models

COPY models/ /app/models/

COPY flair_cache_root/ /app/flair_cache_root/

COPY app.py model_loader.py preload_model.py streamlit_app.py favicon.ico logo.png ./

COPY config/ /app/config/
COPY src/ /app/src/

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000"]