"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


def _check_gpu() -> bool:
    """Check if a CUDA GPU is available via torch."""
    try:
        import torch  # noqa: WPS433

        return torch.cuda.is_available()
    except ImportError:
        return False


@router.get("/api/health")
async def health_check() -> dict:
    """Return service health status and GPU availability."""
    return {
        "status": "ok",
        "gpu_available": _check_gpu(),
    }
