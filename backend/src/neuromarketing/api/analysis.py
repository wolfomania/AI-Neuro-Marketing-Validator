"""Analysis endpoints for video upload and result retrieval."""

import logging
import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from neuromarketing.config import Settings

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _validate_analysis_id(analysis_id: str) -> None:
    if not _UUID_RE.match(analysis_id):
        raise HTTPException(status_code=400, detail="Invalid analysis ID.")
from neuromarketing.schemas.analysis import (
    AnalysisCreateResponse,
    AnalysisStatusResponse,
)
from neuromarketing.schemas.enums import AnalysisStatus
from neuromarketing.services import storage
from neuromarketing.services.pipeline import acquire_gpu_or_reject, run_analysis_pipeline

router = APIRouter(tags=["analysis"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}


@lru_cache
def get_settings() -> Settings:
    """Cached dependency for Settings."""
    return Settings()


@router.post("/api/analysis", status_code=202)
async def create_analysis(
    file: UploadFile,
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> AnalysisCreateResponse:
    """Upload a video file and start analysis in the background.

    Validates the file extension, saves to the upload directory,
    acquires the GPU semaphore, and launches the pipeline as a background task.
    Returns 202 immediately with the analysis_id.
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required.")

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{extension}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    analysis_id = str(uuid.uuid4())
    upload_path = settings.upload_path / f"{analysis_id}{extension}"

    try:
        content = await file.read()
        if len(content) > settings.max_video_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {settings.max_video_size_mb}MB limit.",
            )
        upload_path.write_bytes(content)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to save upload for %s", analysis_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file. Please try again.",
        )

    storage.create_analysis(settings.results_path, analysis_id)

    # Acquire GPU before scheduling background task (returns 429 if busy)
    await acquire_gpu_or_reject()

    # Resolve services from app.state (set during lifespan)
    tribe_service = getattr(request.app.state, "tribe_service", None)
    claude_client = getattr(request.app.state, "claude_client", None)

    if tribe_service is None:
        raise HTTPException(
            status_code=503,
            detail="TRIBE service not initialised. Server may still be starting.",
        )

    background_tasks.add_task(
        run_analysis_pipeline,
        analysis_id=analysis_id,
        video_path=upload_path,
        video_filename=file.filename,
        settings=settings,
        tribe_service=tribe_service,
        claude_client=claude_client,
    )

    return AnalysisCreateResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.QUEUED,
        created_at=datetime.now(timezone.utc),
    )


@router.get("/api/analysis/{analysis_id}/status")
async def get_analysis_status(
    analysis_id: str,
    settings: Settings = Depends(get_settings),
) -> AnalysisStatusResponse:
    """Return the current processing status of an analysis."""
    _validate_analysis_id(analysis_id)
    try:
        status_data = storage.get_status(settings.results_path, analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    return AnalysisStatusResponse(
        analysis_id=status_data["analysis_id"],
        status=status_data["status"],
        stage=status_data.get("stage"),
        stage_label=status_data.get("stage_label"),
        progress_percent=status_data.get("progress_percent", 0),
    )


@router.get("/api/analysis/{analysis_id}")
async def get_analysis_result(
    analysis_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Return the full analysis result.

    Returns the result data for completed or failed analyses.
    If the analysis is still processing, returns 404 with a message.
    """
    _validate_analysis_id(analysis_id)
    result = storage.get_result(settings.results_path, analysis_id)
    if result is None:
        # Check if the analysis exists at all
        try:
            status_data = storage.get_status(settings.results_path, analysis_id)
            status = status_data.get("status", "unknown")
            if status == "failed":
                raise HTTPException(
                    status_code=422,
                    detail="Analysis failed. Check status endpoint for details.",
                )
            raise HTTPException(
                status_code=404,
                detail=f"Analysis result not available. Current status: {status}.",
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Analysis not found.",
            )
    return result


@router.get("/api/analysis/{analysis_id}/images/{image_name}")
async def get_analysis_image(
    analysis_id: str,
    image_name: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve a PNG image from the analysis results directory."""
    _validate_analysis_id(analysis_id)
    if not image_name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Only PNG images are supported.")

    image_path = storage.get_image_path(
        settings.results_path, analysis_id, image_name
    )
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(path=str(image_path), media_type="image/png")
