import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.config import get_settings
from app.database.session import init_db
from app.exceptions import AppError
from app.logging_config import configure_logging
from app.rag.chunker import TextChunker
from app.rag.embedder import EmbeddingService
from app.rag.ollama_client import OllamaClient
from app.rag.pdf_parser import PDFParser
from app.rag.prompt_builder import PromptBuilder
from app.rag.vector_store import FAISSVectorStore

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.faiss_index_file.parent.mkdir(parents=True, exist_ok=True)
    await init_db()
    embedder = EmbeddingService(settings.embedding_model)
    await embedder.initialize()

    vector_store = FAISSVectorStore(
        index_path=settings.faiss_index_file,
        metadata_path=settings.faiss_metadata_file,
    )
    await vector_store.initialize(embedder.dimension)

    app.state.settings = settings
    app.state.pdf_parser = PDFParser()
    app.state.chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.prompt_builder = PromptBuilder()
    app.state.ollama_client = OllamaClient(settings.ollama_base_url, settings.ollama_model)
    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


app.include_router(api_router)
