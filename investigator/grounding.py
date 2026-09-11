from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field


class FindingModel(BaseModel):
    claim: str
    quote: str
    document: str = ""
    chunk_id: str = ""
    status: str = Field(default="supported")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_pool(quote: str, pool: list[str]) -> bool:
    if not quote.strip():
        return False
    raw = quote.strip()
    for blob in pool:
        if raw in blob or normalize(raw) in normalize(blob):
            return True
    return False


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def ground_findings(raw: list[dict], evidence_pool: list[str]) -> list[FindingModel]:
    kept: list[FindingModel] = []
    for item in raw:
        finding = FindingModel.model_validate(item)
        if quote_in_pool(finding.quote, evidence_pool):
            finding.status = "supported"
            kept.append(finding)
        else:
            kept.append(
                FindingModel(
                    claim=finding.claim,
                    quote=finding.quote,
                    document=finding.document,
                    chunk_id=finding.chunk_id,
                    status="unsupported",
                )
            )
    return [item for item in kept if item.status == "supported"][:8]
