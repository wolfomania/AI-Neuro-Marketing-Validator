"""Mock TRIBE v2 service for testing without GPU."""

from pathlib import Path

import numpy as np

from neuromarketing.services.tribe_models import RoiActivation, TribePredictionResult

# Real Glasser atlas ROI names used by TRIBE v2
_GLASSER_ROIS: tuple[tuple[str, str], ...] = (
    ("V1", "Primary Visual Cortex"),
    ("V2", "Secondary Visual Cortex"),
    ("V3", "Third Visual Area"),
    ("V4", "Fourth Visual Area"),
    ("FEF", "Frontal Eye Fields"),
    ("A1", "Primary Auditory Cortex"),
    ("TE1a", "Temporal Area TE1 anterior"),
    ("PHA1", "Para-Hippocampal Area 1"),
    ("EC", "Entorhinal Cortex"),
    ("4", "Primary Motor Cortex"),
    ("3a", "Somatosensory Area 3a"),
    ("3b", "Somatosensory Area 3b"),
    ("5", "Superior Parietal Area 5"),
    ("7", "Superior Parietal Area 7"),
    ("MT", "Middle Temporal Visual Area"),
    ("MST", "Medial Superior Temporal Area"),
    ("LIPv", "Lateral Intraparietal ventral"),
    ("VIP", "Ventral Intraparietal Area"),
    ("8BM", "Prefrontal Area 8BM"),
    ("10v", "Prefrontal Area 10v"),
)

_SEED = 42
_N_TIMESTEPS = 15
_N_VERTICES = 20484
_VIDEO_DURATION_SECONDS = 30.0


class MockTribeService:
    """Mock implementation for testing without GPU."""

    def __init__(self, top_k: int = 20) -> None:
        self._top_k = top_k
        self._loaded = False

    def load_model(self) -> None:
        """Simulate model loading."""
        self._loaded = True

    def predict_activations(
        self, video_path: str, output_dir: Path
    ) -> TribePredictionResult:
        """Return deterministic fake prediction results."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        rng = np.random.default_rng(seed=_SEED)

        timestep_dur = _VIDEO_DURATION_SECONDS / _N_TIMESTEPS

        # Generate deterministic activation values for each ROI
        roi_activations: dict[str, float] = {}
        for roi_name, _ in _GLASSER_ROIS:
            roi_activations[roi_name] = float(
                rng.uniform(0.1, 1.0)
            )

        # Sort by activation to determine ranking
        sorted_rois = sorted(
            roi_activations.items(), key=lambda x: x[1], reverse=True
        )
        all_values = sorted(roi_activations.values())

        def _percentile(value: float) -> float:
            count_below = sum(1 for v in all_values if v < value)
            return round(count_below / len(all_values) * 100, 1)

        # Build top ROIs
        top_rois: list[RoiActivation] = []
        roi_lookup = dict(_GLASSER_ROIS)
        for roi_name, mean_val in sorted_rois[: self._top_k]:
            peak_time = float(rng.uniform(0.0, _VIDEO_DURATION_SECONDS))
            peak_val = mean_val * float(rng.uniform(1.1, 1.5))
            top_rois.append(
                RoiActivation(
                    name=roi_name,
                    full_name=roi_lookup[roi_name],
                    mean_activation=round(mean_val, 4),
                    peak_activation=round(peak_val, 4),
                    peak_time_seconds=round(peak_time, 2),
                    percentile_rank=_percentile(mean_val),
                )
            )

        # Generate placeholder plot files
        plot_paths = _create_placeholder_plots(output_dir)

        # Build segment timestamps
        seg_timestamps = []
        for i in range(_N_TIMESTEPS):
            start = round(i * timestep_dur, 2)
            end = round((i + 1) * timestep_dur, 2)
            seg_timestamps.append((start, end))

        return TribePredictionResult(
            video_path=video_path,
            n_timesteps=_N_TIMESTEPS,
            n_vertices=_N_VERTICES,
            top_rois=tuple(top_rois),
            all_roi_means=roi_activations,
            segment_timestamps=tuple(seg_timestamps),
            timestep_duration_seconds=round(timestep_dur, 3),
            total_duration_seconds=_VIDEO_DURATION_SECONDS,
            plot_paths=plot_paths,
        )


def _create_placeholder_plots(output_dir: Path) -> dict[str, str]:
    """Create minimal 1x1 pixel PNG files as placeholders."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Minimal valid PNG: 1x1 red pixel
    # fmt: off
    minimal_png = (
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx"
        b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    # fmt: on

    views = ["lateral_left", "lateral_right", "medial_left", "medial_right"]
    plot_paths: dict[str, str] = {}

    for view in views:
        path = plots_dir / f"{view}.png"
        path.write_bytes(minimal_png)
        plot_paths[view] = str(path)

    return plot_paths
