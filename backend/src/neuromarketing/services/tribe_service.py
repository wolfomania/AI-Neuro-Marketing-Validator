"""Real TRIBE v2 service implementation. Requires GPU + tribev2 installed."""

from pathlib import Path
from typing import Protocol

from neuromarketing.services.tribe_models import RoiActivation, TribePredictionResult


class BrainPredictionService(Protocol):
    """Abstract interface for brain prediction -- enables mocking."""

    def load_model(self) -> None: ...

    def predict_activations(
        self, video_path: str, output_dir: Path
    ) -> TribePredictionResult: ...


class TribeService:
    """Real implementation using Meta's TRIBE v2. Requires GPU + tribev2."""

    def __init__(
        self, cache_folder: str, hf_token: str, top_k: int = 20
    ) -> None:
        self._cache_folder = cache_folder
        self._hf_token = hf_token
        self._top_k = top_k
        self._model: object | None = None

    def load_model(self) -> None:
        """Load TRIBE v2 model. Call once during app startup."""
        try:
            from tribev2 import TribeModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "tribev2 is not installed. Install from: "
                "git+https://github.com/facebookresearch/tribev2.git"
            ) from exc
        import os

        os.environ["HF_TOKEN"] = self._hf_token
        self._model = TribeModel.from_pretrained(
            "facebook/tribev2", cache_folder=self._cache_folder
        )

    def predict_activations(
        self, video_path: str, output_dir: Path
    ) -> TribePredictionResult:
        """Run full TRIBE v2 prediction pipeline on a video."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import numpy as np
        from tribev2.utils import get_hcp_labels, get_topk_rois  # type: ignore[import-untyped]

        model = self._model

        # Step 1: Get events and predict
        df = model.get_events_dataframe(video_path=video_path)  # type: ignore[union-attr]
        preds, segments = model.predict(events=df)  # type: ignore[union-attr]
        n_timesteps, n_vertices = preds.shape

        # Step 2: Compute per-vertex mean and ROI summaries
        mean_preds = preds.mean(axis=0)
        top_k_raw = get_topk_rois(mean_preds, k=self._top_k)
        hcp_labels = get_hcp_labels()

        # Step 3: Compute all ROI means
        all_roi_means: dict[str, float] = {}
        for roi_name, vertex_indices in hcp_labels.items():
            all_roi_means[roi_name] = float(mean_preds[vertex_indices].mean())

        # Step 4: Compute percentile ranks
        all_values = sorted(all_roi_means.values())

        def _percentile(value: float) -> float:
            count_below = sum(1 for v in all_values if v < value)
            return round(count_below / len(all_values) * 100, 1)

        # Step 5: Estimate timing
        total_duration = (
            len(segments) * 2.0 if segments else n_timesteps * 2.0
        )
        timestep_dur = total_duration / n_timesteps if n_timesteps > 0 else 0.0

        # Step 6: Build RoiActivation list for top K
        top_rois = []
        for roi_name, mean_val in top_k_raw:
            vertex_indices = hcp_labels.get(roi_name, np.array([]))
            if len(vertex_indices) > 0:
                roi_timeseries = preds[:, vertex_indices].mean(axis=1)
                peak_idx = int(np.argmax(roi_timeseries))
                peak_val = float(roi_timeseries[peak_idx])
                peak_time = peak_idx * timestep_dur
            else:
                peak_val = float(mean_val)
                peak_time = 0.0

            top_rois.append(
                RoiActivation(
                    name=roi_name,
                    full_name=roi_name,
                    mean_activation=float(mean_val),
                    peak_activation=peak_val,
                    peak_time_seconds=round(peak_time, 2),
                    percentile_rank=_percentile(float(mean_val)),
                )
            )

        # Step 7: Generate plots
        plot_paths = _generate_plots(mean_preds, output_dir)

        # Step 8: Build segment timestamps
        seg_timestamps = []
        for i in range(n_timesteps):
            start = round(i * timestep_dur, 2)
            end = round((i + 1) * timestep_dur, 2)
            seg_timestamps.append((start, end))

        return TribePredictionResult(
            video_path=video_path,
            n_timesteps=n_timesteps,
            n_vertices=n_vertices,
            top_rois=tuple(top_rois),
            all_roi_means=all_roi_means,
            segment_timestamps=tuple(seg_timestamps),
            timestep_duration_seconds=round(timestep_dur, 3),
            total_duration_seconds=round(total_duration, 2),
            plot_paths=plot_paths,
        )


def _generate_plots(
    mean_preds: "object", output_dir: Path
) -> dict[str, str]:
    """Generate cortical surface map PNGs. Non-fatal on failure."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    views = ["lateral_left", "lateral_right", "medial_left", "medial_right"]
    plot_paths: dict[str, str] = {}

    try:
        import numpy as np
        from nilearn import datasets, plotting  # type: ignore[import-untyped]

        preds_arr = np.asarray(mean_preds)
        fsaverage = datasets.load_fsaverage("fsaverage5")
        n_hemi = len(preds_arr) // 2

        for view in views:
            hemi = "left" if "left" in view else "right"
            view_type = "lateral" if "lateral" in view else "medial"

            mesh = fsaverage[f"pial_{hemi}"]
            sulc = fsaverage[f"sulc_{hemi}"]
            data = preds_arr[:n_hemi] if hemi == "left" else preds_arr[n_hemi:]

            fig = plotting.plot_surf_stat_map(
                mesh,
                data,
                hemi=hemi,
                view=view_type,
                bg_map=sulc,
                colorbar=True,
                cmap="hot",
                title=f"Brain Activation ({view_type} {hemi})",
            )
            path = plots_dir / f"{view}.png"
            fig.savefig(str(path), dpi=150)
            plotting.close_all()
            plot_paths[view] = str(path)
    except Exception:
        # Plotting is non-fatal -- analysis can proceed without it
        for view in views:
            plot_paths[view] = ""

    return plot_paths
