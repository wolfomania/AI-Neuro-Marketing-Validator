"""Pipeline orchestrator: chains video validation, TRIBE prediction, ROI mapping,
Claude analysis, and result storage into a single background task."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from neuromarketing.config import Settings
from neuromarketing.schemas.analysis import (
    AnalysisReport,
    Recommendation,
    RegionInsight,
    TimelineSegment,
)
from neuromarketing.services import storage
from neuromarketing.services.roi_mapper import EnrichedRoi
from neuromarketing.services.tribe_models import TribePredictionResult

logger = logging.getLogger(__name__)

# GPU semaphore -- only one inference at a time
_gpu_semaphore = asyncio.Semaphore(1)


async def acquire_gpu_or_reject() -> None:
    """Try to acquire GPU atomically. Raises HTTPException(429) if busy.

    Uses wait_for with a minimal timeout so that the check and acquire
    happen in a single atomic operation, avoiding TOCTOU races.
    """
    try:
        await asyncio.wait_for(_gpu_semaphore.acquire(), timeout=0.01)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=429,
            detail="GPU is busy processing another video. Try again shortly.",
        )


def release_gpu() -> None:
    """Release the GPU semaphore."""
    _gpu_semaphore.release()


async def run_analysis_pipeline(
    analysis_id: str,
    video_path: Path,
    video_filename: str,
    settings: Settings,
    tribe_service: object,
    claude_client: object | None,
) -> None:
    """Run the full analysis pipeline as a background task.

    Updates status at each stage. Cleans up on completion or failure.
    """
    results_dir = settings.results_path

    try:
        # Stage 1: Validate video
        _update_status(results_dir, analysis_id, "processing", "validating", 5)
        from neuromarketing.utils.video import validate_video

        video_meta = validate_video(
            video_path,
            settings.min_video_duration_seconds,
            settings.max_video_duration_seconds,
        )

        # Stage 2: TRIBE v2 prediction (the heavy GPU part)
        _update_status(results_dir, analysis_id, "processing", "tribe_predict", 15)
        output_dir = results_dir / analysis_id
        prediction: TribePredictionResult = tribe_service.predict_activations(  # type: ignore[union-attr]
            str(video_path), output_dir
        )

        # Stage 3: ROI mapping
        _update_status(results_dir, analysis_id, "processing", "summarizing_rois", 60)
        from neuromarketing.services.roi_mapper import (
            enrich_rois,
            format_for_claude,
            get_category_summary,
            load_glasser_lookup,
        )

        lookup = load_glasser_lookup()
        enriched = enrich_rois(prediction, lookup)
        category_summary = get_category_summary(enriched)
        prediction_summary = format_for_claude(enriched, category_summary, prediction)

        # Stage 4: Claude analysis
        _update_status(
            results_dir, analysis_id, "processing", "analyzing_with_claude", 70
        )

        if claude_client is not None:
            from neuromarketing.services.claude_service import analyze_activations

            report = analyze_activations(
                client=claude_client,
                model=settings.claude_model,
                enriched_rois=enriched,
                category_summary=category_summary,
                prediction_summary=prediction_summary,
                video_filename=video_filename,
                video_duration=prediction.total_duration_seconds,
            )
        else:
            # Mock mode: generate a placeholder report
            report = _generate_mock_report(enriched, prediction)

        # Stage 5: Save results
        _update_status(results_dir, analysis_id, "processing", "saving_results", 90)

        result_data = {
            "analysis_id": analysis_id,
            "status": "completed",
            "created_at": storage.get_status(results_dir, analysis_id).get(
                "created_at", ""
            ),
            "completed_at": _now_iso(),
            "video_metadata": {
                "filename": video_meta.filename,
                "duration_seconds": video_meta.duration_seconds,
                "resolution": video_meta.resolution,
                "file_size_bytes": video_meta.file_size_bytes,
            },
            "report": report.model_dump() if hasattr(report, "model_dump") else report,
            "brain_images": list(prediction.plot_paths.keys()),
        }

        storage.save_result(results_dir, analysis_id, result_data)
        storage.update_status(results_dir, analysis_id, "completed", None, 100)

    except Exception as exc:
        logger.exception("Pipeline failed for analysis %s", analysis_id)
        # Mark as failed
        try:
            storage.update_status(results_dir, analysis_id, "failed", None, 0)
            error_data = {
                "error": str(exc),
                "analysis_id": analysis_id,
                "status": "failed",
            }
            storage.save_result(results_dir, analysis_id, error_data)
        except Exception:
            logger.exception("Failed to save error status for %s", analysis_id)
    finally:
        # Always clean up the uploaded video
        try:
            storage.cleanup_upload(settings.upload_path, analysis_id)
        except Exception:
            logger.warning("Failed to cleanup upload for %s", analysis_id)
        release_gpu()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _update_status(
    results_dir: Path,
    analysis_id: str,
    status: str,
    stage: str | None,
    progress: int,
) -> None:
    """Thin wrapper around storage.update_status with logging."""
    logger.info(
        "Analysis %s: stage=%s progress=%d%%", analysis_id, stage, progress
    )
    storage.update_status(results_dir, analysis_id, status, stage, progress)


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _generate_mock_report(
    enriched_rois: list[EnrichedRoi],
    prediction: TribePredictionResult,
) -> AnalysisReport:
    """Create a placeholder AnalysisReport for testing without Claude API."""
    # Build timeline from segment timestamps
    timeline: list[TimelineSegment] = []
    for i, (start, end) in enumerate(prediction.segment_timestamps[:5]):
        top_names = [r.name for r in enriched_rois[:3]]
        timeline.append(
            TimelineSegment(
                start_seconds=start,
                end_seconds=end,
                label=f"Segment {i + 1}",
                dominant_regions=top_names,
                cognitive_state="engaged" if i % 2 == 0 else "relaxed",
                engagement_level="high" if i % 2 == 0 else "medium",
                insight=f"Mock insight for segment {i + 1}.",
            )
        )

    # Build top region insights
    top_regions: list[RegionInsight] = []
    for rank, roi in enumerate(enriched_rois[:5], start=1):
        top_regions.append(
            RegionInsight(
                roi_name=roi.name,
                full_name=roi.full_name,
                cognitive_function=", ".join(roi.cognitive_functions[:2]),
                activation_rank=rank,
                marketing_implication=roi.marketing_relevance,
            )
        )

    return AnalysisReport(
        executive_summary=(
            "This is a mock analysis report generated without the Claude API. "
            "The video activated visual and auditory processing regions, "
            "suggesting moderate viewer engagement."
        ),
        overall_score=65,
        timeline=timeline,
        top_regions=top_regions,
        recommendations=[
            Recommendation(
                category="visual",
                title="Enhance visual contrast",
                description="Consider increasing colour contrast in key scenes.",
                priority="medium",
            ),
            Recommendation(
                category="audio",
                title="Add emotional music cues",
                description="Background music could boost emotional engagement.",
                priority="high",
            ),
        ],
        strengths=[
            "Strong visual cortex activation indicating good visual content.",
            "Consistent engagement across the video timeline.",
        ],
        weaknesses=[
            "Limited emotional region activation.",
            "No strong memory encoding signals detected.",
        ],
    )
