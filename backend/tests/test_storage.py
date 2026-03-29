"""Tests for storage module-level functions."""

import json
from pathlib import Path

import pytest

from neuromarketing.services import storage


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Provide a temporary results directory."""
    d = tmp_path / "results"
    d.mkdir()
    return d


class TestCreateAnalysis:
    def test_creates_directory_and_status_json(self, results_dir: Path) -> None:
        analysis_id = "test-abc-123"
        storage.create_analysis(results_dir, analysis_id)

        analysis_dir = results_dir / analysis_id
        assert analysis_dir.is_dir()

        status_path = analysis_dir / "status.json"
        assert status_path.exists()

        data = json.loads(status_path.read_text(encoding="utf-8"))
        assert data["analysis_id"] == analysis_id
        assert data["status"] == "queued"
        assert data["progress_percent"] == 0


class TestUpdateStatus:
    def test_updates_status_file(self, results_dir: Path) -> None:
        analysis_id = "test-update-1"
        storage.create_analysis(results_dir, analysis_id)

        storage.update_status(
            results_dir,
            analysis_id,
            status="processing",
            stage="extracting_events",
            progress=15,
        )

        data = storage.get_status(results_dir, analysis_id)
        assert data["status"] == "processing"
        assert data["stage"] == "extracting_events"
        assert data["progress_percent"] == 15


class TestGetStatus:
    def test_reads_status_correctly(self, results_dir: Path) -> None:
        analysis_id = "test-read-1"
        storage.create_analysis(results_dir, analysis_id)

        data = storage.get_status(results_dir, analysis_id)
        assert data["analysis_id"] == analysis_id
        assert data["status"] == "queued"

    def test_raises_for_missing_analysis(self, results_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            storage.get_status(results_dir, "nonexistent-id")


class TestGetResult:
    def test_returns_none_for_missing(self, results_dir: Path) -> None:
        analysis_id = "test-no-result"
        storage.create_analysis(results_dir, analysis_id)

        result = storage.get_result(results_dir, analysis_id)
        assert result is None

    def test_returns_saved_result(self, results_dir: Path) -> None:
        analysis_id = "test-with-result"
        storage.create_analysis(results_dir, analysis_id)

        expected = {"report": "some data", "score": 85}
        storage.save_result(results_dir, analysis_id, expected)

        result = storage.get_result(results_dir, analysis_id)
        assert result == expected
