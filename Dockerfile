# Image Enhancement Studio - runs the Streamlit web app inside Docker.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Train the CNNs for effects 2 and 4 (skipped if models/*.keras already exist)
# and download the pretrained CNNs for effects 1 and 3. Runs once, at build time.
RUN python train_models.py --if-missing

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
