"""Storage operations for analysis results.

Module-level functions only — no classes.
All file operations use pathlib.Path with atomic writes for status updates.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from neuromarketing.schemas.enums import STAGE_LABELS, AnalysisStatus, ProcessingStage


def create_analysis(results_dir: Path, analysis_id: str) -> None:
    """Create results/{id}/ directory and write initial status.json."""
    analysis_dir = results_dir / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    initial_status = {
        "analysis_id": analysis_id,
        "status": AnalysisStatus.QUEUED,
        "stage": None,
        "stage_label": None,
        "progress_percent": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(analysis_dir / "status.json", initial_status)


def update_status(
    results_dir: Path,
    analysis_id: str,
    status: str,
    stage: str | None,
    progress: int,
) -> None:
    """Atomic write to status.json (write to .tmp, then os.replace)."""
    analysis_dir = results_dir / analysis_id
    existing = _read_json(analysis_dir / "status.json")

    stage_label = None
    if stage:
        try:
            stage_label = STAGE_LABELS.get(ProcessingStage(stage))
        except ValueError:
            stage_label = stage

    updated = {
        **existing,
        "status": status,
        "stage": stage,
        "stage_label": stage_label,
        "progress_percent": progress,
    }
    _atomic_write_json(analysis_dir / "status.json", updated)


def get_status(results_dir: Path, analysis_id: str) -> dict:
    """Read status.json for the given analysis."""
    status_path = results_dir / analysis_id / "status.json"
    if not status_path.exists():
        raise FileNotFoundError(
            f"Analysis {analysis_id} not found"
        )
    return _read_json(status_path)


def save_result(results_dir: Path, analysis_id: str, result: dict) -> None:
    """Write result.json for the given analysis."""
    analysis_dir = results_dir / analysis_id
    _atomic_write_json(analysis_dir / "result.json", result)


def get_result(results_dir: Path, analysis_id: str) -> dict | None:
    """Read result.json, returning None if it does not exist."""
    result_path = results_dir / analysis_id / "result.json"
    if not result_path.exists():
        return None
    return _read_json(result_path)


def get_image_path(
    results_dir: Path, analysis_id: str, image_name: str
) -> Path | None:
    """Return path to an image file if it exists, else None.

    Resolves candidate path and verifies it doesn't escape the analysis directory
    (path traversal protection).
    """
    analysis_dir = (results_dir / analysis_id).resolve()
    candidate = (analysis_dir / image_name).resolve()
    if not str(candidate).startswith(str(analysis_dir)):
        return None
    if not candidate.exists():
        return None
    return candidate


def cleanup_upload(upload_dir: Path, analysis_id: str) -> None:
    """Delete uploaded video files matching the analysis_id prefix."""
    for file in upload_dir.iterdir():
        if file.name.startswith(analysis_id):
            file.unlink(missing_ok=True)


# --- Internal helpers ---


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: write to .tmp file, then os.replace."""
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(path))


def _read_json(path: Path) -> dict:
    """Read and parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))
