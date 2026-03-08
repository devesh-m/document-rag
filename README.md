# Document Intelligence Assistant

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.43-red?logo=streamlit)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)](https://ollama.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)

Document Intelligence Assistant is a Retrieval Augmented Generation system for question answering over uploaded PDF documents. The application indexes document chunks with embeddings, retrieves relevant context with FAISS, and generates grounded answers with Ollama.

## Features

- PDF upload and indexing
- Chunk-based retrieval
- Embeddings with `sentence-transformers`
- Vector search with FAISS
- Metadata persistence with SQLite
- Answer generation with Ollama
- FastAPI backend and Streamlit frontend

## Tech Stack

- Backend: FastAPI, SQLAlchemy, SQLite
- Frontend: Streamlit
- Embeddings: sentence-transformers
- Vector store: FAISS
- LLM runtime: Ollama
- Containerization: Docker Compose

## Project Structure

```text
backend/
    app/
        api/
        database/
        rag/
        services/
        config.py
        main.py
frontend/
    streamlit_app.py
storage/
    uploads/
vector_store/
    data/
docker-compose.yml
requirements.txt
```

## API Endpoints

- `GET /health`
- `POST /upload`
- `POST /query`
- `GET /documents`
- `DELETE /documents/{id}`

## Configuration

Copy [.env.example](.env.example) to `.env`.

Key variables:

- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `EMBEDDING_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVAL_TOP_K`
- `STREAMLIT_API_BASE_URL`

## Local Setup

### Prerequisites

- Python 3.11+
- Ollama

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Start the backend

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir backend --reload
```

### Start the frontend

```bash
python -m streamlit run frontend/streamlit_app.py
```

### Access the application

- FastAPI docs: `http://127.0.0.1:8000/docs`
- Streamlit UI: `http://127.0.0.1:8501`

## Docker Setup

Ollama runs on the host machine and is accessed from containers through `host.docker.internal`.

```bash
docker compose up --build
```

### Access the application

- FastAPI docs: `http://127.0.0.1:8000/docs`
- Streamlit UI: `http://127.0.0.1:8501`
