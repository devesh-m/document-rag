from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader


def chunk_text(text: str, *, size: int, overlap: int) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def parse_upload(filename: str, data: bytes, *, chunk_size: int, overlap: int) -> list[dict]:
    pages: list[tuple[int, str]]
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((index, text))
    else:
        pages = [(1, data.decode("utf-8", errors="replace"))]

    chunks: list[dict] = []
    counter = 0
    for page_number, text in pages:
        for piece in chunk_text(text, size=chunk_size, overlap=overlap):
            counter += 1
            chunks.append(
                {
                    "chunk_id": f"{filename}:{page_number}:{counter}",
                    "page": page_number,
                    "text": piece,
                }
            )
    return chunks
