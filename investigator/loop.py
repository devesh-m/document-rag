from __future__ import annotations

import asyncio


def ensure_event_loop() -> None:
    """Gemini/gRPC needs a loop; FastAPI sync routes run in a worker thread without one."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
