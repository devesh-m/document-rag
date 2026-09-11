from __future__ import annotations

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from investigator.db import list_documents
from investigator.vectorstore import PassageStore


def build_tools(store: PassageStore, session: Session):
    @tool
    def search_documents(query: str) -> str:
        """Search indexed passages in Qdrant for a question or keyword."""
        hits = store.search(query)
        if not hits:
            return "No matching passages."
        lines = []
        for hit in hits:
            lines.append(
                f"[{hit['chunk_id']}] {hit['filename']} p.{hit['page']} score={hit['score']:.3f}\n{hit['text']}"
            )
        return "\n\n".join(lines)

    @tool
    def read_passage(chunk_id: str) -> str:
        """Read the full stored text of one passage by chunk_id."""
        hit = store.get_chunk(chunk_id)
        if hit is None:
            return f"No passage stored for {chunk_id}."
        return f"[{hit['chunk_id']}] {hit['filename']} p.{hit['page']}\n{hit['text']}"

    @tool
    def list_library() -> str:
        """List documents currently in the library."""
        rows = list_documents(session)
        if not rows:
            return "The library is empty. Upload a document first."
        return "\n".join(
            f"id={row.id} {row.filename} pages={row.page_count} chunks={row.chunk_count}"
            for row in rows
        )

    return [search_documents, read_passage, list_library]
