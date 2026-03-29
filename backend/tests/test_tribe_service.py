"""Tests for TRIBE v2 service layer using the mock implementation."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from neuromarketing.services.mock_tribe import MockTribeService
from neuromarketing.services.tribe_models import RoiActivation, TribePredictionResult
from neuromarketing.services.tribe_service import BrainPredictionService


@pytest.fixture
def mock_tribe() -> MockTribeService:
    """Provide a loaded MockTribeService instance."""
    svc = MockTribeService(top_k=20)
    svc.load_model()
    return svc


@pytest.fixture
def prediction(mock_tribe: MockTribeService, tmp_path: Path) -> TribePredictionResult:
    """Run a mock prediction and return the result."""
    return mock_tribe.predict_activations(
        video_path="/tmp/test_video.mp4", output_dir=tmp_path
    )


class TestTribePredictionResultShape:
    """Verify the structure of the prediction result."""

    def test_n_timesteps(self, prediction: TribePredictionResult) -> None:
        assert prediction.n_timesteps == 15

    def test_n_vertices(self, prediction: TribePredictionResult) -> None:
        assert prediction.n_vertices == 20484

    def test_video_path(self, prediction: TribePredictionResult) -> None:
        assert prediction.video_path == "/tmp/test_video.mp4"

    def test_total_duration(self, prediction: TribePredictionResult) -> None:
        assert prediction.total_duration_seconds == 30.0

    def test_timestep_duration(self, prediction: TribePredictionResult) -> None:
        assert prediction.timestep_duration_seconds == 2.0

    def test_segment_timestamps_count(
        self, prediction: TribePredictionResult
    ) -> None:
        assert len(prediction.segment_timestamps) == 15

    def test_segment_timestamps_range(
        self, prediction: TribePredictionResult
    ) -> None:
        first = prediction.segment_timestamps[0]
        last = prediction.segment_timestamps[-1]
        assert first[0] == 0.0
        assert last[1] == 30.0


class TestTopRois:
    """Verify top ROI output."""

    def test_top_rois_count(self, prediction: TribePredictionResult) -> None:
        assert len(prediction.top_rois) == 20

    def test_top_rois_are_roi_activation(
        self, prediction: TribePredictionResult
    ) -> None:
        for roi in prediction.top_rois:
            assert isinstance(roi, RoiActivation)

    def test_top_rois_sorted_descending(
        self, prediction: TribePredictionResult
    ) -> None:
        activations = [r.mean_activation for r in prediction.top_rois]
        assert activations == sorted(activations, reverse=True)

    def test_roi_has_full_name(
        self, prediction: TribePredictionResult
    ) -> None:
        for roi in prediction.top_rois:
            assert len(roi.full_name) > 0


class TestAllRoiMeans:
    """Verify the all_roi_means dictionary."""

    def test_has_expected_keys(
        self, prediction: TribePredictionResult
    ) -> None:
        expected = {
            "V1", "V2", "V3", "V4", "FEF", "A1", "TE1a", "PHA1", "EC",
            "4", "3a", "3b", "5", "7", "MT", "MST", "LIPv", "VIP",
            "8BM", "10v",
        }
        assert set(prediction.all_roi_means.keys()) == expected

    def test_values_are_positive(
        self, prediction: TribePredictionResult
    ) -> None:
        for value in prediction.all_roi_means.values():
            assert value > 0.0


class TestPercentileRank:
    """Verify percentile rank constraints."""

    def test_percentile_in_range(
        self, prediction: TribePredictionResult
    ) -> None:
        for roi in prediction.top_rois:
            assert 0.0 <= roi.percentile_rank <= 100.0

    def test_highest_activation_has_high_percentile(
        self, prediction: TribePredictionResult
    ) -> None:
        # The top ROI should have a high percentile
        top = prediction.top_rois[0]
        assert top.percentile_rank >= 90.0


class TestPlotPaths:
    """Verify plot file generation."""

    def test_contains_all_views(
        self, prediction: TribePredictionResult
    ) -> None:
        expected_views = {
            "lateral_left", "lateral_right",
            "medial_left", "medial_right",
        }
        assert set(prediction.plot_paths.keys()) == expected_views

    def test_plot_files_exist(
        self, prediction: TribePredictionResult
    ) -> None:
        for path_str in prediction.plot_paths.values():
            assert Path(path_str).exists()

    def test_plot_files_are_png(
        self, prediction: TribePredictionResult
    ) -> None:
        for path_str in prediction.plot_paths.values():
            assert path_str.endswith(".png")


class TestImmutability:
    """Verify frozen dataclasses cannot be mutated."""

    def test_prediction_result_is_frozen(
        self, prediction: TribePredictionResult
    ) -> None:
        with pytest.raises(FrozenInstanceError):
            prediction.n_timesteps = 99  # type: ignore[misc]

    def test_roi_activation_is_frozen(
        self, prediction: TribePredictionResult
    ) -> None:
        roi = prediction.top_rois[0]
        with pytest.raises(FrozenInstanceError):
            roi.mean_activation = 999.0  # type: ignore[misc]


class TestProtocolConformance:
    """Verify MockTribeService satisfies BrainPredictionService protocol."""

    def test_mock_satisfies_protocol(self) -> None:
        svc: BrainPredictionService = MockTribeService()
        assert hasattr(svc, "load_model")
        assert hasattr(svc, "predict_activations")

    def test_mock_callable_as_protocol(self, tmp_path: Path) -> None:
        svc: BrainPredictionService = MockTribeService()
        svc.load_model()
        result = svc.predict_activations("/tmp/video.mp4", tmp_path)
        assert isinstance(result, TribePredictionResult)


class TestModelNotLoaded:
    """Verify error when model is not loaded."""

    def test_predict_before_load_raises(self, tmp_path: Path) -> None:
        svc = MockTribeService()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            svc.predict_activations("/tmp/video.mp4", tmp_path)


class TestDeterminism:
    """Verify mock produces deterministic output."""

    def test_two_runs_produce_same_result(self, tmp_path: Path) -> None:
        svc1 = MockTribeService()
        svc1.load_model()
        r1 = svc1.predict_activations("/tmp/v.mp4", tmp_path / "run1")

        svc2 = MockTribeService()
        svc2.load_model()
        r2 = svc2.predict_activations("/tmp/v.mp4", tmp_path / "run2")

        assert r1.all_roi_means == r2.all_roi_means
        assert [r.mean_activation for r in r1.top_rois] == [
            r.mean_activation for r in r2.top_rois
        ]
