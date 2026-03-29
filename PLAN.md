# AI Neuro-Marketing Validator — Implementation Plan (APPROVED)

## Overview
Upload marketing video → TRIBE v2 predicts brain activations → aggregate to brain regions → Claude API interprets → display results.

## Tech Stack
- Backend: FastAPI (Python 3.10+), uv
- Frontend: Vite + React (TypeScript)
- ML: TRIBE v2 (Meta FAIR) — `github.com/facebookresearch/tribev2`
- LLM: Claude API (Anthropic SDK)
- Brain viz: tribev2 built-in plotting (static PNGs)

## Phase Order
```
Phase 0: TRIBE v2 Bootstrap [HARD GATE]
Phase 1: Project Skeleton
Phase 2 || Phase 6: Upload+Validation || Glasser Atlas Seed (parallel)
Phase 3: TRIBE v2 Service
Phase 4: Claude Integration
Phase 5: Pipeline Orchestration
Phase 7: Frontend Integration
```

## Phase 0: TRIBE v2 Bootstrap
0.1 Clone + install tribev2, pin commit SHA in pyproject.toml
0.2 Verify dependency compatibility with FastAPI
0.3 Run minimal predict + plotting, dump fixture shapes
0.4 FastAPI co-residence smoke test
Exit criteria: fixture shapes dumped, 4 PNG plots generated, FastAPI smoke passes

## Phase 1: Project Skeleton
- Backend: FastAPI app factory, config (pydantic-settings), health endpoint
- Frontend: Vite + React shell
- Storage: module-level functions (not class)
- Upload limit: 100MB in FastAPI config

## Phase 2: Upload + Validation
- POST /api/analysis endpoint (multipart upload)
- ffprobe duration check (15-60s), reject with 422
- storage.py: temp paths, cleanup_job() in finally blocks

## Phase 3: TRIBE v2 Service
- TribeService with Protocol-based interface
- TribePredictionResult (frozen dataclass): top_rois, all_roi_means, segments
- Aggregation: mean, peak, peak_time per ROI (top 20 for Claude)
- Mock fixture for GPU-free testing

## Phase 4: Claude Integration
- tool_use for structured output (AnalysisReport schema)
- Fallback: code-fence JSON extraction → Pydantic validate → 1 retry
- @pytest.mark.live_api for real API tests

## Phase 5: Pipeline Orchestration
- GPU semaphore: asyncio.wait_for(sem.acquire(), timeout=0) → 429 on busy
- _generate_plots() via tribev2.plotting
- Pipeline runner with finally-block cleanup
- Wire to API endpoints

## Phase 6: Glasser Atlas Seed
- Seed script: get_hcp_labels() → Claude generates descriptions → manual review
- Startup validation: compare JSON keys against tribev2 labels

## Phase 7: Frontend
- Upload flow with VideoPreview
- ResultsView: brain viz PNG grid, SummaryReport, Timeline, TopRegionsTable, Recommendations
- Polling with useAnalysis hook (3s interval)

## Key Data Structures
```python
@dataclass(frozen=True)
class RoiActivation:
    name: str
    mean_activation: float
    peak_activation: float
    peak_time_seconds: float

@dataclass(frozen=True)
class TribePredictionResult:
    video_path: str
    n_timesteps: int
    n_vertices: int  # 20484
    top_rois: list[RoiActivation]
    all_roi_means: dict[str, float]
    segments: list[dict]
    timestep_duration_seconds: float
    total_duration_seconds: float
```

## API Contracts
- POST /api/analysis → 202 {analysis_id, status: "queued"}
- GET /api/analysis/{id}/status → {status, stage, progress_percent}
- GET /api/analysis/{id} → full result with report
- GET /api/analysis/{id}/images/{name} → PNG
- GET /api/health → {status, tribe_model_loaded, gpu_available}

## Level 3 (A/B) Seams
- TribePredictionResult is self-contained per video
- Pipeline splits into run_prediction() + run_analysis()
- Claude service gains compare() method
- API gains POST /api/comparison
- Frontend components take data as props (composable)
