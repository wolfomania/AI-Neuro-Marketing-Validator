"""Tests for video validation utilities."""

from pathlib import Path
from unittest.mock import patch

import pytest

from neuromarketing.utils.video import VideoMetadata, validate_video


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    """Create a dummy file to act as a video."""
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"\x00" * 1024)
    return video


def _mock_probe_result(duration: float = 30.0, width: int = 1920, height: int = 1080) -> dict:
    """Build a fake ffprobe result."""
    return {
        "format": {"duration": str(duration)},
        "streams": [
            {
                "codec_type": "video",
                "width": width,
                "height": height,
            }
        ],
    }


class TestValidateVideo:
    @patch("neuromarketing.utils.video.ffmpeg.probe")
    def test_valid_video_returns_metadata(
        self, mock_probe: object, fake_video: Path
    ) -> None:
        mock_probe.return_value = _mock_probe_result(duration=30.0)

        result = validate_video(fake_video, min_duration=15, max_duration=60)

        assert isinstance(result, VideoMetadata)
        assert result.filename == "sample.mp4"
        assert result.duration_seconds == 30.0
        assert result.resolution == "1920x1080"
        assert result.file_size_bytes == 1024

    @patch("neuromarketing.utils.video.ffmpeg.probe")
    def test_duration_too_short_raises(
        self, mock_probe: object, fake_video: Path
    ) -> None:
        mock_probe.return_value = _mock_probe_result(duration=5.0)

        with pytest.raises(ValueError, match="too short"):
            validate_video(fake_video, min_duration=15, max_duration=60)

    @patch("neuromarketing.utils.video.ffmpeg.probe")
    def test_duration_too_long_raises(
        self, mock_probe: object, fake_video: Path
    ) -> None:
        mock_probe.return_value = _mock_probe_result(duration=120.0)

        with pytest.raises(ValueError, match="too long"):
            validate_video(fake_video, min_duration=15, max_duration=60)

    @patch("neuromarketing.utils.video.ffmpeg.probe")
    def test_frozen_dataclass_is_immutable(
        self, mock_probe: object, fake_video: Path
    ) -> None:
        mock_probe.return_value = _mock_probe_result(duration=30.0)

        result = validate_video(fake_video, min_duration=15, max_duration=60)

        with pytest.raises(AttributeError):
            result.filename = "changed.mp4"  # type: ignore[misc]
