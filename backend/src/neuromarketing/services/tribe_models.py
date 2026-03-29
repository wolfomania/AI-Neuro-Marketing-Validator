"""Core data models for TRIBE v2 brain prediction results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RoiActivation:
    """Single ROI's aggregated activation data."""

    name: str  # Glasser atlas ROI name, e.g. "V1", "FFC"
    full_name: str  # Human-readable, e.g. "Primary Visual Cortex"
    mean_activation: float  # Mean across all timesteps
    peak_activation: float  # Maximum activation value
    peak_time_seconds: float  # Timestamp of peak activation
    percentile_rank: float  # Rank within this video (0-100, 100 = highest)


@dataclass(frozen=True)
class TribePredictionResult:
    """Immutable container for all TRIBE v2 outputs for ONE video."""

    video_path: str
    n_timesteps: int
    n_vertices: int  # Expected ~20484
    top_rois: tuple[RoiActivation, ...]  # Top K ROIs by mean activation
    all_roi_means: dict[str, float]  # All ROI name -> mean activation
    segment_timestamps: tuple[tuple[float, float], ...]  # (start, end) per timestep
    timestep_duration_seconds: float
    total_duration_seconds: float
    plot_paths: dict[str, str]  # view_name -> PNG file path
