from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from investigator.config import Settings, get_settings
from investigator.db import get_session, init_db, list_documents
from investigator.graph import run_investigation
from investigator.services import ingest_file, record_investigation, serialize_document
from investigator.vectorstore import PassageStore


@lru_cache(maxsize=1)
def get_store() -> PassageStore:
    from investigator.loop import ensure_event_loop

    ensure_event_loop()
    return PassageStore(get_settings())


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Doc Investigator", version="0.2.0", lifespan=lifespan)
settings = get_settings()
_origins = settings.cors_origin_list
_allow_all = "*" in _origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else _origins,
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)


def db_session() -> Session:
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) or exc.__class__.__name__},
    )


def index_file(session: Session, config: Settings, *, filename: str, data: bytes):
    try:
        store = get_store()
        return ingest_file(session, store, config, filename=filename, data=data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not index the document: {exc}") from exc


class InvestigateIn(BaseModel):
    brief: str = Field(min_length=12)


SAMPLE_BRIEF = """Check the vendor policy for:
1. How long is customer data retained?
2. Is encryption at rest required?
3. Can vendors access production without a ticket?
"""


@app.get("/")
def root() -> dict:
    return {"app": "Doc Investigator", "health": "/health", "ui": "Deploy the React app on Vercel."}


@app.get("/health")
def health(config: Settings = Depends(get_settings)) -> dict:
    return {"status": "ok", "app": config.app_name, "env": config.app_env}


@app.get("/api/sample")
def sample() -> dict:
    settings = get_settings()
    policy = ""
    if settings.demo_policy_path.exists():
        policy = settings.demo_policy_path.read_text(encoding="utf-8")
    return {"brief": SAMPLE_BRIEF.strip(), "policy": policy}


@app.get("/api/sample-file")
def sample_file() -> FileResponse:
    settings = get_settings()
    if not settings.demo_policy_path.exists():
        raise HTTPException(status_code=404, detail="Sample policy is missing.")
    return FileResponse(
        settings.demo_policy_path,
        filename="Acme-Vendor-Security-Policy.txt",
        media_type="text/plain",
    )


@app.get("/api/documents")
def documents(session: Session = Depends(db_session)) -> dict:
    return {"documents": [serialize_document(item) for item in list_documents(session)]}


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    session: Session = Depends(db_session),
    config: Settings = Depends(get_settings),
) -> dict:
    filename = file.filename or "upload.txt"
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    row = index_file(session, config, filename=filename, data=data)
    session.flush()
    return serialize_document(row)


@app.post("/api/seed")
def seed_sample(
    session: Session = Depends(db_session),
    config: Settings = Depends(get_settings),
) -> dict:
    path = config.demo_policy_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample policy is missing.")
    row = index_file(session, config, filename=path.name, data=path.read_bytes())
    session.flush()
    return serialize_document(row)


@app.post("/api/investigate")
def investigate(
    payload: InvestigateIn,
    session: Session = Depends(db_session),
    config: Settings = Depends(get_settings),
) -> dict:
    if not list_documents(session):
        raise HTTPException(status_code=400, detail="Upload or seed a document first.")
    try:
        store = get_store()
        result = run_investigation(config, store, session, payload.brief.strip())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Investigation failed: {exc}") from exc
    row = record_investigation(session, payload.brief.strip(), result)
    session.flush()
    return {
        "id": row.id,
        "brief": row.brief,
        "summary": row.summary,
        "status": row.status,
        "step_count": row.step_count,
        "trace": result.get("trace") or [],
        "findings": result.get("findings") or [],
    }
