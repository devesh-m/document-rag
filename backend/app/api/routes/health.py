from fastapi import APIRouter, Request

from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
    )
