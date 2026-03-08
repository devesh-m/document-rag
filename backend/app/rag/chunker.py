from dataclasses import dataclass

from app.rag.pdf_parser import ParsedPage


@dataclass(slots=True)
class TextChunk:
    text: str
    chunk_order: int
    page_number: int | None


class TextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(self, pages: list[ParsedPage]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_order = 0

        for page in pages:
            start = 0
            text = " ".join(page.text.split())
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunk_order += 1
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            chunk_order=chunk_order,
                            page_number=page.page_number,
                        )
                    )
                if end >= len(text):
                    break
                start = end - self.chunk_overlap

        return chunks
