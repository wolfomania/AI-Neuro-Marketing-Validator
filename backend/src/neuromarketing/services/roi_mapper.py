"""ROI enrichment: maps Glasser atlas regions to cognitive functions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from neuromarketing.services.tribe_models import RoiActivation, TribePredictionResult

logger = logging.getLogger(__name__)

_DEFAULT_LOOKUP_PATH = Path(__file__).resolve().parent.parent / "data" / "glasser_lookup.json"

_UNKNOWN_ENTRY: dict = {
    "full_name": "Unknown Region",
    "cognitive_functions": ["General cortical processing"],
    "category": "executive",
    "emotional_valence": "neutral",
    "marketing_relevance": "This region contributes to general cortical processing",
}


@dataclass(frozen=True)
class EnrichedRoi:
    """ROI activation enriched with cognitive function metadata."""

    name: str
    full_name: str
    mean_activation: float
    peak_activation: float
    peak_time_seconds: float
    percentile_rank: float
    cognitive_functions: list[str]
    category: str
    emotional_valence: str
    marketing_relevance: str


def load_glasser_lookup(lookup_path: Path | None = None) -> dict[str, dict]:
    """Load the Glasser lookup JSON.

    Args:
        lookup_path: Optional override path. Defaults to data/glasser_lookup.json.

    Returns:
        Dict mapping ROI short name to its metadata dict.
    """
    path = lookup_path or _DEFAULT_LOOKUP_PATH
    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, dict] = json.load(fh)
    logger.info("Loaded Glasser lookup with %d entries from %s", len(data), path)
    return data


def enrich_rois(
    prediction: TribePredictionResult,
    lookup: dict[str, dict],
) -> list[EnrichedRoi]:
    """Enrich top ROIs with cognitive function labels from the lookup table.

    For ROIs not found in the lookup, reasonable defaults are used.
    """
    enriched: list[EnrichedRoi] = []
    for roi in prediction.top_rois:
        entry = lookup.get(roi.name, _UNKNOWN_ENTRY)
        enriched.append(
            EnrichedRoi(
                name=roi.name,
                full_name=entry.get("full_name", roi.full_name),
                mean_activation=roi.mean_activation,
                peak_activation=roi.peak_activation,
                peak_time_seconds=roi.peak_time_seconds,
                percentile_rank=roi.percentile_rank,
                cognitive_functions=list(entry.get("cognitive_functions", [])),
                category=entry.get("category", "executive"),
                emotional_valence=entry.get("emotional_valence", "neutral"),
                marketing_relevance=entry.get(
                    "marketing_relevance",
                    "This region contributes to general cortical processing",
                ),
            )
        )
    return enriched


def get_category_summary(enriched_rois: list[EnrichedRoi]) -> dict[str, float]:
    """Aggregate mean activation by category.

    Returns:
        Dict mapping category name to average mean_activation across ROIs
        in that category.  Example: {"visual": 0.85, "emotional": 0.72}
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for roi in enriched_rois:
        sums[roi.category] = sums.get(roi.category, 0.0) + roi.mean_activation
        counts[roi.category] = counts.get(roi.category, 0) + 1
    return {cat: sums[cat] / counts[cat] for cat in sorted(sums)}


def format_for_claude(
    enriched_rois: list[EnrichedRoi],
    category_summary: dict[str, float],
    prediction: TribePredictionResult,
) -> str:
    """Format enriched ROI data as a structured text block for the Claude prompt.

    Returns:
        Markdown-formatted string summarising video metadata,
        category-level activation, and top regions.
    """
    lines: list[str] = [
        "## Brain Activation Analysis",
        "",
        "### Video Metadata",
        f"- **Video**: {prediction.video_path}",
        f"- **Duration**: {prediction.total_duration_seconds:.1f}s",
        f"- **Timesteps**: {prediction.n_timesteps}",
        f"- **Vertices**: {prediction.n_vertices}",
        "",
        "### Category-Level Activation Summary",
        "",
    ]

    for cat in sorted(category_summary, key=category_summary.get, reverse=True):  # type: ignore[arg-type]
        lines.append(f"- **{cat.capitalize()}**: {category_summary[cat]:.3f}")

    lines += ["", "### Top Activated Regions", ""]

    for rank, roi in enumerate(enriched_rois[:20], start=1):
        lines.append(
            f"{rank}. **{roi.name}** ({roi.full_name}) — "
            f"mean {roi.mean_activation:.3f}, peak {roi.peak_activation:.3f} "
            f"@ {roi.peak_time_seconds:.1f}s, "
            f"percentile {roi.percentile_rank:.0f}"
        )
        lines.append(f"   - Functions: {', '.join(roi.cognitive_functions)}")
        lines.append(f"   - Category: {roi.category} | Valence: {roi.emotional_valence}")
        lines.append(f"   - Marketing insight: {roi.marketing_relevance}")
        lines.append("")

    return "\n".join(lines)
