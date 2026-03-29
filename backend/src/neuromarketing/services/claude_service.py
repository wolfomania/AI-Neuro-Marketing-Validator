"""Claude API analysis service for neuro-marketing brain activation data."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict

from anthropic import Anthropic

from neuromarketing.schemas.analysis import AnalysisReport
from neuromarketing.services.roi_mapper import EnrichedRoi

logger = logging.getLogger(__name__)

_MAX_RETRIES = 1


def create_claude_client(api_key: str) -> Anthropic:
    """Create an Anthropic client instance."""
    if not api_key:
        raise ValueError("Anthropic API key must not be empty")
    return Anthropic(api_key=api_key)


def analyze_activations(
    client: Anthropic,
    model: str,
    enriched_rois: list[EnrichedRoi],
    category_summary: dict[str, float],
    prediction_summary: str,
    video_filename: str,
    video_duration: float,
) -> AnalysisReport:
    """Send enriched ROI data to Claude and return a structured AnalysisReport.

    Strategy:
    1. Primary: Use tool_use with AnalysisReport JSON schema
    2. Fallback: Parse JSON from text response
    3. Retry once on parse failure
    """
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(
        enriched_rois, category_summary, prediction_summary,
        video_filename, video_duration,
    )
    tool = _build_analysis_tool_schema()

    for attempt in range(1 + _MAX_RETRIES):
        messages = [{"role": "user", "content": user_prompt}]
        if attempt > 0:
            messages[0]["content"] = (
                user_prompt
                + "\n\nIMPORTANT: Your previous response could not be parsed. "
                "You MUST call the generate_analysis_report tool with valid JSON."
            )

        logger.info("Claude analysis attempt %d/%d", attempt + 1, 1 + _MAX_RETRIES)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=[tool],
        )

        # Try tool_use first
        try:
            return _parse_tool_response(response)
        except (ValueError, KeyError, IndexError) as exc:
            logger.warning("Tool parse failed (attempt %d): %s", attempt + 1, exc)

        # Fallback: extract from text blocks
        text_parts = [
            block.text for block in response.content
            if hasattr(block, "text")
        ]
        if text_parts:
            try:
                return _parse_text_response("\n".join(text_parts))
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("Text parse failed (attempt %d): %s", attempt + 1, exc)

    raise RuntimeError(
        "Failed to parse Claude response after retries. "
        "Neither tool_use nor text fallback produced a valid AnalysisReport."
    )


def _build_system_prompt() -> str:
    """System prompt establishing Claude as a neuro-marketing analyst."""
    return (
        "You are a senior neuro-marketing strategist who reads brain activation data "
        "to evaluate advertising and video content effectiveness.\n\n"
        "## Your Expertise\n"
        "You translate brain encoding model predictions into actionable marketing "
        "insights. The data you receive comes from a computational neuroscience model "
        "(TRIBE v2) that predicts how the human brain responds to video stimuli, "
        "region by region, across the cortical surface.\n\n"
        "## Communication Style\n"
        "- Write for marketing directors and brand managers, NOT neuroscientists.\n"
        "- Replace jargon with plain business language "
        '(e.g., say "emotional resonance" instead of "amygdala activation").\n'
        "- Be specific: name the exact moments, scenes, and creative elements.\n"
        "- Be actionable: every insight should suggest what to do next.\n\n"
        "## Analysis Requirements\n"
        "1. **Executive Summary**: 2-3 sentences capturing the video's overall "
        "neuro-marketing effectiveness.\n"
        "2. **Overall Score**: A single 1-100 score reflecting how effectively the "
        "video engages viewers neurologically (50 = average ad, 80+ = exceptional).\n"
        "3. **Timeline Segments**: Identify 3-6 meaningful scenes based on shifts in "
        "brain activation patterns. Do NOT create one segment per timestep. "
        "Group adjacent timesteps with similar activation profiles. "
        "Assign engagement_level as 'high', 'medium', or 'low' based on the overall "
        "activation intensity during that segment.\n"
        "4. **Top Regions**: Rank the most activated brain regions by marketing "
        "relevance and explain what each means for the creative.\n"
        "5. **Recommendations**: Provide 3-5 concrete, prioritized suggestions to "
        "improve the video's neurological impact. Use marketing language.\n"
        "6. **Strengths & Weaknesses**: List 2-4 of each.\n\n"
        "## Output Format\n"
        "You MUST call the generate_analysis_report tool with your analysis. "
        "Do NOT return plain text."
    )


def _build_analysis_tool_schema() -> dict:
    """JSON schema for the AnalysisReport, used as a tool definition."""
    return {
        "name": "generate_analysis_report",
        "description": (
            "Generate a structured neuro-marketing analysis report "
            "from brain activation data."
        ),
        "input_schema": {
            "type": "object",
            "required": [
                "executive_summary",
                "overall_score",
                "timeline",
                "top_regions",
                "recommendations",
                "strengths",
                "weaknesses",
            ],
            "properties": {
                "executive_summary": {
                    "type": "string",
                    "description": "2-3 sentence overview of the video's neuro-marketing effectiveness.",
                },
                "overall_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Effectiveness score (1-100). 50 = average, 80+ = exceptional.",
                },
                "timeline": {
                    "type": "array",
                    "description": "3-6 meaningful timeline segments.",
                    "items": {
                        "type": "object",
                        "required": [
                            "start_seconds", "end_seconds", "label",
                            "dominant_regions", "cognitive_state",
                            "engagement_level", "insight",
                        ],
                        "properties": {
                            "start_seconds": {"type": "number"},
                            "end_seconds": {"type": "number"},
                            "label": {"type": "string", "description": "Short scene label."},
                            "dominant_regions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Most active brain regions in this segment.",
                            },
                            "cognitive_state": {
                                "type": "string",
                                "description": "What the viewer is likely experiencing.",
                            },
                            "engagement_level": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                            "insight": {
                                "type": "string",
                                "description": "Marketing-relevant observation about this segment.",
                            },
                        },
                    },
                },
                "top_regions": {
                    "type": "array",
                    "description": "Top activated brain regions ranked by marketing relevance.",
                    "items": {
                        "type": "object",
                        "required": [
                            "roi_name", "full_name", "cognitive_function",
                            "activation_rank", "marketing_implication",
                        ],
                        "properties": {
                            "roi_name": {"type": "string"},
                            "full_name": {"type": "string"},
                            "cognitive_function": {"type": "string"},
                            "activation_rank": {"type": "integer", "minimum": 1},
                            "marketing_implication": {"type": "string"},
                        },
                    },
                },
                "recommendations": {
                    "type": "array",
                    "description": "3-5 prioritized marketing recommendations.",
                    "items": {
                        "type": "object",
                        "required": ["category", "title", "description", "priority"],
                        "properties": {
                            "category": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                    },
                },
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 strengths of the video.",
                },
                "weaknesses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 weaknesses of the video.",
                },
            },
        },
    }


def _build_user_prompt(
    enriched_rois: list[EnrichedRoi],
    category_summary: dict[str, float],
    prediction_summary: str,
    video_filename: str,
    video_duration: float,
) -> str:
    """Build the user message with all brain activation data."""
    roi_data = "\n".join(
        f"- {roi.name} ({roi.full_name}): mean={roi.mean_activation:.3f}, "
        f"peak={roi.peak_activation:.3f} @ {roi.peak_time_seconds:.1f}s, "
        f"percentile={roi.percentile_rank:.0f}, "
        f"functions=[{', '.join(roi.cognitive_functions)}], "
        f"category={roi.category}, valence={roi.emotional_valence}"
        for roi in enriched_rois
    )

    category_lines = "\n".join(
        f"- {cat}: {score:.3f}" for cat, score in category_summary.items()
    )

    return (
        f"# Brain Activation Analysis Request\n\n"
        f"**Video**: {video_filename}\n"
        f"**Duration**: {video_duration:.1f} seconds\n\n"
        f"## Category-Level Activation Summary\n{category_lines}\n\n"
        f"## Top Activated Brain Regions\n{roi_data}\n\n"
        f"## Detailed Prediction Summary\n{prediction_summary}\n\n"
        f"Please analyze this brain activation data and call the "
        f"generate_analysis_report tool with your complete analysis."
    )


def _parse_tool_response(response: object) -> AnalysisReport:
    """Extract AnalysisReport from Claude's tool_use response."""
    for block in response.content:  # type: ignore[attr-defined]
        if block.type == "tool_use" and block.name == "generate_analysis_report":
            return AnalysisReport.model_validate(block.input)
    raise ValueError("No generate_analysis_report tool_use block found in response")


def _parse_text_response(text: str) -> AnalysisReport:
    """Fallback: extract JSON from markdown code fences in text response."""
    pattern = r"```(?:json)?\s*(\{.*\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError("No JSON code fence found in text response")
    data = json.loads(match.group(1))
    return AnalysisReport.model_validate(data)
