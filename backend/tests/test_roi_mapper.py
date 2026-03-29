"""Tests for the ROI mapper module."""

from __future__ import annotations

import pytest

from neuromarketing.services.roi_mapper import (
    EnrichedRoi,
    enrich_rois,
    format_for_claude,
    get_category_summary,
    load_glasser_lookup,
)
from neuromarketing.services.tribe_models import RoiActivation, TribePredictionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_roi(
    name: str = "V1",
    full_name: str = "Primary Visual Cortex",
    mean: float = 0.85,
    peak: float = 0.95,
    peak_t: float = 3.2,
    pct: float = 98.0,
) -> RoiActivation:
    return RoiActivation(
        name=name,
        full_name=full_name,
        mean_activation=mean,
        peak_activation=peak,
        peak_time_seconds=peak_t,
        percentile_rank=pct,
    )


def _make_prediction(
    rois: tuple[RoiActivation, ...] | None = None,
) -> TribePredictionResult:
    default_rois = (
        _make_roi("V1", "Primary Visual Cortex", 0.90, 0.98, 2.0, 99),
        _make_roi("FFC", "Fusiform Face Complex", 0.82, 0.91, 4.5, 95),
        _make_roi("A1", "Primary Auditory Cortex", 0.75, 0.88, 1.0, 90),
        _make_roi("OFC", "Orbitofrontal Cortex", 0.70, 0.85, 6.0, 85),
        _make_roi("H", "Hippocampus", 0.65, 0.80, 5.0, 80),
        _make_roi("44", "Broca's Area", 0.60, 0.78, 3.0, 75),
        _make_roi("FEF", "Frontal Eye Fields", 0.55, 0.72, 7.0, 70),
        _make_roi("46", "DLPFC", 0.50, 0.68, 8.0, 65),
    )
    return TribePredictionResult(
        video_path="/tmp/test_video.mp4",
        n_timesteps=30,
        n_vertices=20484,
        top_rois=rois if rois is not None else default_rois,
        all_roi_means={"V1": 0.90, "FFC": 0.82, "A1": 0.75},
        segment_timestamps=((0.0, 1.0), (1.0, 2.0)),
        timestep_duration_seconds=1.0,
        total_duration_seconds=30.0,
        plot_paths={"lateral": "/tmp/lateral.png"},
    )


@pytest.fixture()
def lookup() -> dict[str, dict]:
    return load_glasser_lookup()


@pytest.fixture()
def prediction() -> TribePredictionResult:
    return _make_prediction()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadGlasserLookup:
    def test_loads_and_has_minimum_entries(self, lookup: dict) -> None:
        assert len(lookup) >= 100

    def test_known_region_has_required_keys(self, lookup: dict) -> None:
        v1 = lookup["V1"]
        for key in ("full_name", "cognitive_functions", "category", "emotional_valence", "marketing_relevance"):
            assert key in v1, f"Missing key: {key}"

    def test_categories_are_valid(self, lookup: dict) -> None:
        valid = {
            "visual", "auditory", "motor", "emotional", "attention",
            "memory", "language", "social", "executive", "somatosensory", "spatial",
        }
        for name, entry in lookup.items():
            assert entry["category"] in valid, f"{name} has invalid category: {entry['category']}"


class TestEnrichRois:
    def test_returns_enriched_list(self, lookup: dict, prediction: TribePredictionResult) -> None:
        enriched = enrich_rois(prediction, lookup)
        assert len(enriched) == len(prediction.top_rois)
        assert all(isinstance(r, EnrichedRoi) for r in enriched)

    def test_enriched_fields_populated(self, lookup: dict, prediction: TribePredictionResult) -> None:
        enriched = enrich_rois(prediction, lookup)
        first = enriched[0]
        assert first.name == "V1"
        assert first.mean_activation == 0.90
        assert len(first.cognitive_functions) >= 1
        assert first.category in (
            "visual", "auditory", "motor", "emotional", "attention",
            "memory", "language", "social", "executive", "somatosensory", "spatial",
        )

    def test_handles_unknown_roi(self, lookup: dict) -> None:
        unknown_roi = _make_roi("ZZZZZ", "Totally Unknown", 0.5, 0.6, 1.0, 50)
        pred = _make_prediction(rois=(unknown_roi,))
        enriched = enrich_rois(pred, lookup)
        assert len(enriched) == 1
        assert enriched[0].full_name == "Unknown Region"
        assert enriched[0].category == "executive"

    def test_handles_empty_rois(self, lookup: dict) -> None:
        pred = _make_prediction(rois=())
        enriched = enrich_rois(pred, lookup)
        assert enriched == []


class TestGetCategorySummary:
    def test_returns_expected_categories(self, lookup: dict, prediction: TribePredictionResult) -> None:
        enriched = enrich_rois(prediction, lookup)
        summary = get_category_summary(enriched)
        assert isinstance(summary, dict)
        assert len(summary) >= 1
        assert all(isinstance(v, float) for v in summary.values())

    def test_single_category(self) -> None:
        rois = [
            EnrichedRoi(
                name="V1", full_name="V1", mean_activation=0.8,
                peak_activation=0.9, peak_time_seconds=1.0,
                percentile_rank=95, cognitive_functions=["a"],
                category="visual", emotional_valence="neutral",
                marketing_relevance="x",
            ),
            EnrichedRoi(
                name="V2", full_name="V2", mean_activation=0.6,
                peak_activation=0.7, peak_time_seconds=2.0,
                percentile_rank=80, cognitive_functions=["b"],
                category="visual", emotional_valence="neutral",
                marketing_relevance="y",
            ),
        ]
        summary = get_category_summary(rois)
        assert "visual" in summary
        assert abs(summary["visual"] - 0.7) < 1e-9

    def test_empty_input(self) -> None:
        assert get_category_summary([]) == {}


class TestFormatForClaude:
    def test_returns_nonempty_string(self, lookup: dict, prediction: TribePredictionResult) -> None:
        enriched = enrich_rois(prediction, lookup)
        summary = get_category_summary(enriched)
        result = format_for_claude(enriched, summary, prediction)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_key_information(self, lookup: dict, prediction: TribePredictionResult) -> None:
        enriched = enrich_rois(prediction, lookup)
        summary = get_category_summary(enriched)
        result = format_for_claude(enriched, summary, prediction)
        assert "Brain Activation Analysis" in result
        assert "Video Metadata" in result
        assert "test_video.mp4" in result
        assert "V1" in result
        assert "Category-Level Activation Summary" in result
        assert "Top Activated Regions" in result

    def test_limits_to_20_regions(self, lookup: dict) -> None:
        rois = tuple(
            _make_roi(f"R{i}", f"Region {i}", 0.5, 0.6, 1.0, 50)
            for i in range(25)
        )
        pred = _make_prediction(rois=rois)
        enriched = enrich_rois(pred, lookup)
        summary = get_category_summary(enriched)
        result = format_for_claude(enriched, summary, pred)
        # Should have entries numbered 1-20, not 21+
        assert "20." in result
        assert "21." not in result


class TestEnrichedRoiFrozen:
    def test_is_frozen(self) -> None:
        roi = EnrichedRoi(
            name="V1", full_name="V1", mean_activation=0.8,
            peak_activation=0.9, peak_time_seconds=1.0,
            percentile_rank=95, cognitive_functions=["a"],
            category="visual", emotional_valence="neutral",
            marketing_relevance="x",
        )
        with pytest.raises(AttributeError):
            roi.name = "V2"  # type: ignore[misc]
