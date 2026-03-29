"""Video validation utilities using ffmpeg-python."""

from dataclasses import dataclass
from pathlib import Path

import ffmpeg


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable container for video file metadata."""

    filename: str
    duration_seconds: float
    resolution: str
    file_size_bytes: int


def validate_video(
    file_path: Path,
    min_duration: int,
    max_duration: int,
) -> VideoMetadata:
    """Validate video file and extract metadata using ffprobe.

    Args:
        file_path: Path to the video file.
        min_duration: Minimum allowed duration in seconds.
        max_duration: Maximum allowed duration in seconds.

    Returns:
        VideoMetadata with extracted information.

    Raises:
        ValueError: If duration is outside allowed bounds.
        RuntimeError: If ffprobe fails to read the file.
    """
    try:
        probe = ffmpeg.probe(str(file_path))
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"Failed to read video file: {file_path.name}. "
            "Ensure the file is a valid video."
        ) from e

    video_streams = [
        s for s in probe.get("streams", []) if s["codec_type"] == "video"
    ]
    if not video_streams:
        raise ValueError(
            f"No video stream found in {file_path.name}. "
            "Please upload a valid video file."
        )

    video_stream = video_streams[0]
    duration = float(probe["format"].get("duration", 0))
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    file_size = file_path.stat().st_size

    if duration < min_duration:
        raise ValueError(
            f"Video is too short ({duration:.1f}s). "
            f"Minimum duration is {min_duration} seconds."
        )

    if duration > max_duration:
        raise ValueError(
            f"Video is too long ({duration:.1f}s). "
            f"Maximum duration is {max_duration} seconds."
        )

    return VideoMetadata(
        filename=file_path.name,
        duration_seconds=duration,
        resolution=f"{width}x{height}",
        file_size_bytes=file_size,
    )
