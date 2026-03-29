"""Tests for the pipeline orchestrator."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neuromarketing.config import Settings
from neuromarketing.services.mock_tribe import MockTribeService
from neuromarketing.services.pipeline import (
    _generate_mock_report,
    _gpu_semaphore,
    acquire_gpu_or_reject,
    release_gpu,
    run_analysis_pipeline,
)
from neuromarketing.services import storage


@pytest.fixture
def pipeline_settings(tmp_path: Path) -> Settings:
    """Settings with temporary directories for pipeline tests."""
    return Settings(
        anthropic_api_key="",
        hf_token="",
        upload_dir=str(tmp_path / "uploads"),
        results_dir=str(tmp_path / "results"),
        use_mock=True,
        min_video_duration_seconds=1,
        max_video_duration_seconds=120,
    )


@pytest.fixture
def loaded_tribe_service() -> MockTribeService:
    """A loaded MockTribeService."""
    svc = MockTribeService(top_k=20)
    svc.load_model()
    return svc


@pytest.fixture
def fake_video(pipeline_settings: Settings) -> tuple[Path, str]:
    """Create a fake video file in the upload directory."""
    pipeline_settings.upload_path.mkdir(parents=True, exist_ok=True)
    analysis_id = "test-analysis-001"
    video_path = pipeline_settings.upload_path / f"{analysis_id}.mp4"
    video_path.write_bytes(b"fake video content")
    return video_path, analysis_id


@pytest.fixture(autouse=True)
async def reset_gpu_semaphore():
    """Ensure GPU semaphore is released between tests."""
    # Force reset semaphore to 1 available before each test
    while _gpu_semaphore.locked():
        _gpu_semaphore.release()
    yield
    # Reset again after
    while _gpu_semaphore.locked():
        _gpu_semaphore.release()


def _make_mock_video_meta() -> MagicMock:
    """Create a mock VideoMetadata object."""
    meta = MagicMock()
    meta.filename = "test.mp4"
    meta.duration_seconds = 30.0
    meta.resolution = "1920x1080"
    meta.file_size_bytes = 1024
    return meta


class TestRunAnalysisPipeline:
    """Tests for the full pipeline execution."""

    @pytest.mark.asyncio
    async def test_pipeline_success_mock_mode(
        self,
        pipeline_settings: Settings,
        loaded_tribe_service: MockTribeService,
        fake_video: tuple[Path, str],
    ) -> None:
        """Pipeline runs successfully with MockTribeService and no Claude."""
        video_path, analysis_id = fake_video
        storage.create_analysis(pipeline_settings.results_path, analysis_id)

        # Acquire semaphore manually since pipeline releases it in finally
        await _gpu_semaphore.acquire()

        with patch(
            "neuromarketing.utils.video.validate_video",
            return_value=_make_mock_video_meta(),
        ):
            await run_analysis_pipeline(
                analysis_id=analysis_id,
                video_path=video_path,
                video_filename="test.mp4",
                settings=pipeline_settings,
                tribe_service=loaded_tribe_service,
                claude_client=None,
            )

        result = storage.get_result(pipeline_settings.results_path, analysis_id)
        assert result is not None
        assert result["status"] == "completed"
        assert result["analysis_id"] == analysis_id
        assert "report" in result
        assert "video_metadata" in result
        assert result["video_metadata"]["filename"] == "test.mp4"

    @pytest.mark.asyncio
    async def test_pipeline_updates_status_through_stages(
        self,
        pipeline_settings: Settings,
        loaded_tribe_service: MockTribeService,
        fake_video: tuple[Path, str],
    ) -> None:
        """Pipeline updates status at each stage."""
        video_path, analysis_id = fake_video
        storage.create_analysis(pipeline_settings.results_path, analysis_id)

        recorded_stages: list[str | None] = []
        original_update = storage.update_status

        def tracking_update(
            results_dir: Path,
            aid: str,
            status: str,
            stage: str | None,
            progress: int,
        ) -> None:
            recorded_stages.append(stage)
            original_update(results_dir, aid, status, stage, progress)

        await _gpu_semaphore.acquire()

        with (
            patch(
                "neuromarketing.utils.video.validate_video",
                return_value=_make_mock_video_meta(),
            ),
            patch(
                "neuromarketing.services.pipeline.storage.update_status",
                side_effect=tracking_update,
            ),
        ):
            await run_analysis_pipeline(
                analysis_id=analysis_id,
                video_path=video_path,
                video_filename="test.mp4",
                settings=pipeline_settings,
                tribe_service=loaded_tribe_service,
                claude_client=None,
            )

        expected_stages = [
            "validating",
            "tribe_predict",
            "summarizing_rois",
            "analyzing_with_claude",
            "saving_results",
            None,  # final "completed" update has stage=None
        ]
        assert recorded_stages == expected_stages

    @pytest.mark.asyncio
    async def test_pipeline_marks_failed_on_exception(
        self,
        pipeline_settings: Settings,
        loaded_tribe_service: MockTribeService,
        fake_video: tuple[Path, str],
    ) -> None:
        """Pipeline marks status as 'failed' when an exception occurs."""
        video_path, analysis_id = fake_video
        storage.create_analysis(pipeline_settings.results_path, analysis_id)

        await _gpu_semaphore.acquire()

        with patch(
            "neuromarketing.utils.video.validate_video",
            side_effect=ValueError("Video too short"),
        ):
            await run_analysis_pipeline(
                analysis_id=analysis_id,
                video_path=video_path,
                video_filename="test.mp4",
                settings=pipeline_settings,
                tribe_service=loaded_tribe_service,
                claude_client=None,
            )

        status_data = storage.get_status(pipeline_settings.results_path, analysis_id)
        assert status_data["status"] == "failed"

        result = storage.get_result(pipeline_settings.results_path, analysis_id)
        assert result is not None
        assert result["status"] == "failed"
        assert "Video too short" in result["error"]

    @pytest.mark.asyncio
    async def test_pipeline_cleans_up_uploaded_video(
        self,
        pipeline_settings: Settings,
        loaded_tribe_service: MockTribeService,
        fake_video: tuple[Path, str],
    ) -> None:
        """Pipeline removes uploaded video file in the finally block."""
        video_path, analysis_id = fake_video
        storage.create_analysis(pipeline_settings.results_path, analysis_id)

        assert video_path.exists(), "Video file should exist before pipeline"

        await _gpu_semaphore.acquire()

        with patch(
            "neuromarketing.utils.video.validate_video",
            return_value=_make_mock_video_meta(),
        ):
            await run_analysis_pipeline(
                analysis_id=analysis_id,
                video_path=video_path,
                video_filename="test.mp4",
                settings=pipeline_settings,
                tribe_service=loaded_tribe_service,
                claude_client=None,
            )

        assert not video_path.exists(), "Video file should be cleaned up after pipeline"


class TestGpuSemaphore:
    """Tests for GPU semaphore behaviour."""

    @pytest.mark.asyncio
    async def test_acquire_gpu_or_reject_raises_429_when_busy(self) -> None:
        """acquire_gpu_or_reject raises HTTPException(429) when GPU is busy."""
        from fastapi import HTTPException

        # Acquire semaphore to simulate GPU busy
        await _gpu_semaphore.acquire()

        try:
            with pytest.raises(HTTPException) as exc_info:
                await acquire_gpu_or_reject()
            assert exc_info.value.status_code == 429
            assert "GPU is busy" in str(exc_info.value.detail)
        finally:
            _gpu_semaphore.release()

    @pytest.mark.asyncio
    async def test_acquire_gpu_succeeds_when_free(self) -> None:
        """acquire_gpu_or_reject succeeds when GPU is available."""
        await acquire_gpu_or_reject()
        # Clean up
        release_gpu()

    @pytest.mark.asyncio
    async def test_release_gpu_frees_semaphore(self) -> None:
        """release_gpu makes the semaphore available again."""
        await _gpu_semaphore.acquire()
        release_gpu()
        # Should succeed now
        await acquire_gpu_or_reject()
        release_gpu()


class TestGenerateMockReport:
    """Tests for the mock report generator."""

    def test_mock_report_structure(
        self, mock_tribe_service: MockTribeService
    ) -> None:
        """Mock report has all required fields."""
        from neuromarketing.services.roi_mapper import (
            enrich_rois,
            load_glasser_lookup,
        )

        output_dir = Path("__test_mock_report")
        output_dir.mkdir(exist_ok=True)
        try:
            prediction = mock_tribe_service.predict_activations(
                "fake.mp4", output_dir
            )
            lookup = load_glasser_lookup()
            enriched = enrich_rois(prediction, lookup)

            report = _generate_mock_report(enriched, prediction)

            assert report.executive_summary
            assert 1 <= report.overall_score <= 100
            assert len(report.timeline) > 0
            assert len(report.top_regions) > 0
            assert len(report.recommendations) > 0
            assert len(report.strengths) > 0
            assert len(report.weaknesses) > 0
        finally:
            import shutil

            shutil.rmtree(output_dir, ignore_errors=True)
