import httpx

from app.exceptions import ExternalServiceError


class OllamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                )
                response.raise_for_status()
                payload = response.json()
                return payload.get("response", "")
        except httpx.HTTPError as exc:
            raise ExternalServiceError(f"Ollama request failed: {exc}") from exc
