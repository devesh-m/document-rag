from app.schemas import CitationResponse


class PromptBuilder:
    def build(self, question: str, citations: list[CitationResponse]) -> str:
        context_blocks: list[str] = []
        for index, citation in enumerate(citations, start=1):
            page_label = f"page {citation.page_number}" if citation.page_number is not None else "page unknown"
            context_blocks.append(
                f"[C{index}] File: {citation.filename} | {page_label}\n{citation.snippet}"
            )

        joined_context = "\n\n".join(context_blocks)
        return (
            "You are a document intelligence assistant. Answer only from the provided context. "
            "If the answer cannot be found in the context, say that the available documents do not contain enough evidence. "
            "When you use a source, cite it inline using labels like [C1] or [C2].\n\n"
            f"Context:\n{joined_context}\n\n"
            f"Question: {question}\n\n"
            "Return a concise but complete answer grounded in the context."
        )
