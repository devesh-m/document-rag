# Doc Investigator

This is no longer chat-with-PDF. It is a **LangGraph investigator** over a document library.

You upload a policy (or seed the sample). You paste a brief. The agent must call tools (`list_library`, `search_documents`, `read_passage`) against **Qdrant**. Findings whose quotes are not in tool output are dropped.

Embeddings and chat both use the **Gemini API**. There is no Ollama, FAISS, or sentence-transformers.

## Stack

- LangGraph agent loop with LangChain tools
- Gemini chat + Gemini embeddings
- Qdrant vector database
- PostgreSQL for document metadata
- FastAPI + React

## Run

```bash
cp .env.example .env
docker compose up --build
```

API: `http://127.0.0.1:8002/health`

```bash
cd apps/web
npm install
npm run dev
```

UI: `http://127.0.0.1:5175`

Set `GEMINI_API_KEY` in `.env`.

## Deploy

- **API:** Render blueprint in `render.yaml`. Set `GEMINI_API_KEY`, Neon `DATABASE_URL` (`postgresql+psycopg://...`), and Qdrant Cloud `QDRANT_URL` + `QDRANT_API_KEY`.
- **UI:** Vercel on `apps/web`. Set `VITE_API_BASE_URL` to the Render URL (no trailing slash).

## Demo

1. Click **Seed sample policy** (retention 18 months, encryption required, vendor access needs a ticket).
2. Click **Load sample brief** (three checks; one should be refused if the quote is missing).
3. Click **Run agent**. Read the findings and the tool trace.

## API

| Method | Path |
| --- | --- |
| `GET` | `/health` |
| `GET` | `/api/sample` |
| `GET` | `/api/documents` |
| `POST` | `/api/documents` |
| `POST` | `/api/seed` |
| `POST` | `/api/investigate` |
