"""Tests for the Claude analysis service (all mocked, no real API calls)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from neuromarketing.schemas.analysis import AnalysisReport
from neuromarketing.services.claude_service import (
    _build_analysis_tool_schema,
    _build_system_prompt,
    _build_user_prompt,
    _parse_text_response,
    _parse_tool_response,
    analyze_activations,
    create_claude_client,
)
from neuromarketing.services.roi_mapper import EnrichedRoi


# --- Fixtures ---


def _make_enriched_roi(**overrides: object) -> EnrichedRoi:
    defaults = {
        "name": "V1",
        "full_name": "Primary Visual Cortex",
        "mean_activation": 0.85,
        "peak_activation": 1.2,
        "peak_time_seconds": 3.5,
        "percentile_rank": 95.0,
        "cognitive_functions": ["Visual processing", "Edge detection"],
        "category": "visual",
        "emotional_valence": "neutral",
        "marketing_relevance": "Strong visual attention to the creative",
    }
    defaults.update(overrides)
    return EnrichedRoi(**defaults)


SAMPLE_REPORT_DICT = {
    "executive_summary": "The video strongly engages visual and emotional processing.",
    "overall_score": 72,
    "timeline": [
        {
            "start_seconds": 0.0,
            "end_seconds": 5.0,
            "label": "Opening hook",
            "dominant_regions": ["V1", "FFC"],
            "cognitive_state": "Visual attention capture",
            "engagement_level": "high",
            "insight": "Strong opening that grabs viewer attention.",
        },
        {
            "start_seconds": 5.0,
            "end_seconds": 12.0,
            "label": "Product reveal",
            "dominant_regions": ["PHA1"],
            "cognitive_state": "Memory encoding",
            "engagement_level": "medium",
            "insight": "Product association is forming but could be stronger.",
        },
        {
            "start_seconds": 12.0,
            "end_seconds": 15.0,
            "label": "Call to action",
            "dominant_regions": ["V1"],
            "cognitive_state": "Decision processing",
            "engagement_level": "low",
            "insight": "The CTA does not drive sufficient engagement.",
        },
    ],
    "top_regions": [
        {
            "roi_name": "V1",
            "full_name": "Primary Visual Cortex",
            "cognitive_function": "Visual processing",
            "activation_rank": 1,
            "marketing_implication": "Viewers are visually engaged with the content.",
        },
    ],
    "recommendations": [
        {
            "category": "Creative",
            "title": "Strengthen the closing CTA",
            "description": "Add emotional imagery to the last 3 seconds.",
            "priority": "high",
        },
    ],
    "strengths": ["Strong visual hook", "Good emotional resonance in mid-section"],
    "weaknesses": ["Weak CTA", "Low memory encoding signals"],
}


def _make_tool_use_response(report_dict: dict | None = None) -> SimpleNamespace:
    """Build a mock Claude response with a tool_use content block."""
    data = report_dict or SAMPLE_REPORT_DICT
    tool_block = SimpleNamespace(
        type="tool_use",
        name="generate_analysis_report",
        input=data,
    )
    return SimpleNamespace(content=[tool_block])


def _make_text_response(report_dict: dict | None = None) -> SimpleNamespace:
    """Build a mock Claude response with a text content block containing JSON."""
    data = report_dict or SAMPLE_REPORT_DICT
    json_text = f"Here is my analysis:\n```json\n{json.dumps(data)}\n```"
    text_block = SimpleNamespace(type="text", text=json_text)
    return SimpleNamespace(content=[text_block])


# --- Unit tests for prompt builders ---


class TestBuildSystemPrompt:
    def test_returns_non_empty_string(self) -> None:
        prompt = _build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_key_terms(self) -> None:
        prompt = _build_system_prompt()
        for term in [
            "neuro-marketing",
            "brain activation",
            "marketing",
            "timeline",
            "engagement_level",
            "generate_analysis_report",
        ]:
            assert term.lower() in prompt.lower(), f"Missing term: {term}"

    def test_discourages_jargon(self) -> None:
        prompt = _build_system_prompt()
        assert "jargon" in prompt.lower()


class TestBuildUserPrompt:
    def test_includes_video_info(self) -> None:
        roi = _make_enriched_roi()
        prompt = _build_user_prompt(
            enriched_rois=[roi],
            category_summary={"visual": 0.85},
            prediction_summary="## Brain Activation Analysis",
            video_filename="test_ad.mp4",
            video_duration=15.0,
        )
        assert "test_ad.mp4" in prompt
        assert "15.0" in prompt

    def test_includes_roi_data(self) -> None:
        roi = _make_enriched_roi(name="FFC", full_name="Fusiform Face Complex")
        prompt = _build_user_prompt(
            enriched_rois=[roi],
            category_summary={},
            prediction_summary="",
            video_filename="video.mp4",
            video_duration=10.0,
        )
        assert "FFC" in prompt
        assert "Fusiform Face Complex" in prompt

    def test_includes_category_summary(self) -> None:
        prompt = _build_user_prompt(
            enriched_rois=[],
            category_summary={"visual": 0.85, "emotional": 0.72},
            prediction_summary="",
            video_filename="video.mp4",
            video_duration=10.0,
        )
        assert "visual" in prompt
        assert "0.850" in prompt
        assert "emotional" in prompt


class TestBuildAnalysisToolSchema:
    def test_returns_valid_schema(self) -> None:
        schema = _build_analysis_tool_schema()
        assert schema["name"] == "generate_analysis_report"
        assert "input_schema" in schema

    def test_has_required_fields(self) -> None:
        schema = _build_analysis_tool_schema()
        required = schema["input_schema"]["required"]
        for field in [
            "executive_summary", "overall_score", "timeline",
            "top_regions", "recommendations", "strengths", "weaknesses",
        ]:
            assert field in required, f"Missing required field: {field}"

    def test_timeline_items_have_engagement_level(self) -> None:
        schema = _build_analysis_tool_schema()
        timeline_props = schema["input_schema"]["properties"]["timeline"]["items"]["properties"]
        assert "engagement_level" in timeline_props
        assert timeline_props["engagement_level"]["enum"] == ["high", "medium", "low"]


# --- Unit tests for response parsers ---


class TestParseToolResponse:
    def test_parses_valid_tool_response(self) -> None:
        response = _make_tool_use_response()
        report = _parse_tool_response(response)
        assert isinstance(report, AnalysisReport)
        assert report.overall_score == 72
        assert len(report.timeline) == 3

    def test_raises_on_missing_tool_block(self) -> None:
        response = SimpleNamespace(content=[
            SimpleNamespace(type="text", text="No tool here"),
        ])
        with pytest.raises(ValueError, match="No generate_analysis_report"):
            _parse_tool_response(response)


class TestParseTextResponse:
    def test_parses_json_from_code_fence(self) -> None:
        text = f"Analysis:\n```json\n{json.dumps(SAMPLE_REPORT_DICT)}\n```"
        report = _parse_text_response(text)
        assert isinstance(report, AnalysisReport)
        assert report.overall_score == 72

    def test_parses_without_json_language_tag(self) -> None:
        text = f"Analysis:\n```\n{json.dumps(SAMPLE_REPORT_DICT)}\n```"
        report = _parse_text_response(text)
        assert isinstance(report, AnalysisReport)

    def test_raises_on_no_json(self) -> None:
        with pytest.raises(ValueError, match="No JSON code fence"):
            _parse_text_response("No JSON here at all.")

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _parse_text_response("```json\n{invalid json}\n```")


# --- Integration tests (mocked API) ---


class TestAnalyzeActivations:
    def _build_mock_client(self, responses: list[SimpleNamespace]) -> MagicMock:
        client = MagicMock()
        client.messages.create.side_effect = responses
        return client

    def test_success_with_tool_use(self) -> None:
        client = self._build_mock_client([_make_tool_use_response()])
        report = analyze_activations(
            client=client,
            model="claude-sonnet-4-20250514",
            enriched_rois=[_make_enriched_roi()],
            category_summary={"visual": 0.85},
            prediction_summary="Test summary",
            video_filename="ad.mp4",
            video_duration=15.0,
        )
        assert isinstance(report, AnalysisReport)
        assert report.overall_score == 72
        assert client.messages.create.call_count == 1

    def test_fallback_to_text_parsing(self) -> None:
        """When tool_use block is absent, falls back to text JSON extraction."""
        text_response = _make_text_response()
        client = self._build_mock_client([text_response])
        report = analyze_activations(
            client=client,
            model="claude-sonnet-4-20250514",
            enriched_rois=[_make_enriched_roi()],
            category_summary={"visual": 0.85},
            prediction_summary="Test summary",
            video_filename="ad.mp4",
            video_duration=15.0,
        )
        assert isinstance(report, AnalysisReport)
        assert report.overall_score == 72

    def test_retry_on_first_failure(self) -> None:
        """First call returns garbage, second call succeeds."""
        bad_response = SimpleNamespace(content=[
            SimpleNamespace(type="text", text="I cannot comply."),
        ])
        good_response = _make_tool_use_response()
        client = self._build_mock_client([bad_response, good_response])

        report = analyze_activations(
            client=client,
            model="claude-sonnet-4-20250514",
            enriched_rois=[_make_enriched_roi()],
            category_summary={"visual": 0.85},
            prediction_summary="Test summary",
            video_filename="ad.mp4",
            video_duration=15.0,
        )
        assert isinstance(report, AnalysisReport)
        assert client.messages.create.call_count == 2

    def test_raises_after_all_retries_exhausted(self) -> None:
        bad_response = SimpleNamespace(content=[
            SimpleNamespace(type="text", text="Sorry, no data."),
        ])
        client = self._build_mock_client([bad_response, bad_response])

        with pytest.raises(RuntimeError, match="Failed to parse Claude response"):
            analyze_activations(
                client=client,
                model="claude-sonnet-4-20250514",
                enriched_rois=[_make_enriched_roi()],
                category_summary={"visual": 0.85},
                prediction_summary="Test summary",
                video_filename="ad.mp4",
                video_duration=15.0,
            )
        assert client.messages.create.call_count == 2


class TestCreateClaudeClient:
    def test_raises_on_empty_key(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            create_claude_client("")

    @patch("neuromarketing.services.claude_service.Anthropic")
    def test_creates_client(self, mock_anthropic_cls: MagicMock) -> None:
        client = create_claude_client("sk-test-key")
        mock_anthropic_cls.assert_called_once_with(api_key="sk-test-key")
        assert client is mock_anthropic_cls.return_value
