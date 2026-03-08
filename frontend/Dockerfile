FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY frontend/requirements.docker.txt ./requirements.docker.txt
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY frontend ./frontend

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
