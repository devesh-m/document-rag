from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass(slots=True)
class ParsedPage:
    page_number: int
    text: str


class PDFParser:
    def parse(self, file_bytes: bytes) -> list[ParsedPage]:
        reader = PdfReader(BytesIO(file_bytes))
        pages: list[ParsedPage] = []

        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(ParsedPage(page_number=index, text=text))

        return pages
