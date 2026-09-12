from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from investigator.config import Settings
from investigator.db import Document, Finding, Investigation, get_document, list_documents
from investigator.ingest import parse_upload
from investigator.vectorstore import PassageStore


def serialize_document(row: Document) -> dict:
    return {
        "id": row.id,
        "filename": row.filename,
        "page_count": row.page_count,
        "chunk_count": row.chunk_count,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def ingest_file(
    session: Session,
    store: PassageStore,
    settings: Settings,
    *,
    filename: str,
    data: bytes,
) -> Document:
    chunks = parse_upload(
        filename,
        data,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from that file.")
    pages = max(item["page"] for item in chunks)
    row = Document(filename=filename, page_count=pages, chunk_count=len(chunks))
    session.add(row)
    session.flush()
    store.upsert_chunks(document_id=row.id, filename=filename, chunks=chunks)
    return row


def record_investigation(session: Session, brief: str, result: dict) -> Investigation:
    row = Investigation(
        brief=brief,
        summary=result.get("summary") or "",
        status="done",
        step_count=int(result.get("step_count") or 0),
    )
    session.add(row)
    session.flush()
    for item in result.get("findings") or []:
        session.add(
            Finding(
                investigation_id=row.id,
                claim=item["claim"],
                quote=item["quote"],
                document=item.get("document") or "",
                chunk_id=item.get("chunk_id") or "",
                status=item.get("status") or "supported",
            )
        )
    return row


def remove_document(session: Session, store: PassageStore, document_id: int) -> None:
    row = get_document(session, document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    store.delete_document(row.id)
    session.delete(row)


def clear_library(session: Session, store: PassageStore) -> int:
    rows = list_documents(session)
    for row in rows:
        store.delete_document(row.id)
        session.delete(row)
    return len(rows)
