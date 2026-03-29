# AI Neuro-Marketing Validator

Predict how the human brain responds to marketing videos using neuroscience-grade AI.

Upload a video &rarr; **TRIBE v2** (Meta FAIR) predicts cortical activations &rarr; regions are mapped to the **Glasser Brain Atlas** &rarr; **Claude AI** interprets the neuroscience data into actionable marketing insights.

---

## Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌───────────────┐
│  React + TS  │────▶│  FastAPI (Python)     │────▶│  TRIBE v2     │
│  (Vite)      │◀────│                       │◀────│  (Meta FAIR)  │
└──────────────┘     │  Pipeline:            │     └───────────────┘
                     │  1. Video validation   │
                     │  2. Brain prediction   │     ┌───────────────┐
                     │  3. ROI aggregation    │────▶│  Claude API   │
                     │  4. AI interpretation  │◀────│  (Anthropic)  │
                     └──────────────────────┘     └───────────────┘
```

## Features

- **Video Upload & Validation** — accepts 15-60s marketing videos (up to 100 MB), validates with ffprobe
- **Brain Activation Prediction** — TRIBE v2 predicts neural responses across 20,484 cortical vertices
- **Region-of-Interest Mapping** — aggregates vertex-level data to Glasser Atlas brain regions (180 per hemisphere)
- **AI-Powered Interpretation** — Claude analyzes top activated regions and generates marketing insights
- **Real-time Processing Status** — live progress tracking through each pipeline stage
- **Visual Brain Maps** — static brain activation heatmaps via TRIBE v2 plotting
- **Mock Mode** — full development workflow without GPU or API keys

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript, Vite 8 |
| Backend | FastAPI, Python 3.11+, uv |
| ML Model | TRIBE v2 (Meta FAIR) |
| LLM | Claude API (Anthropic SDK) |
| Brain Atlas | Glasser HCP MMP 1.0 (360 regions) |

## Quick Start

### Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- Node.js 18+
- (Optional) NVIDIA GPU with CUDA for TRIBE v2 inference
- (Optional) Anthropic API key for Claude analysis

### Setup

```bash
# Clone
git clone https://github.com/wolfomania/AI-Neuro-Marketing-Validator.git
cd AI-Neuro-Marketing-Validator

# Backend
cd backend
cp ../.env.example .env  # edit with your API keys
uv sync --all-extras
uv run uvicorn neuromarketing.main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:5173` with the API at `http://localhost:8000`.

### Mock Mode (no GPU required)

Set `NM_USE_MOCK=true` in your `.env` to run the full pipeline with synthetic brain data — perfect for frontend development and testing.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analysis` | Upload video, start analysis |
| `GET` | `/api/analysis/{id}/status` | Poll processing status |
| `GET` | `/api/analysis/{id}` | Get full results with report |
| `GET` | `/api/analysis/{id}/images/{name}` | Brain activation PNGs |
| `GET` | `/api/health` | Service health check |

## Project Structure

```
.
├── backend/
│   ├── src/neuromarketing/
│   │   ├── api/            # FastAPI routes
│   │   ├── schemas/        # Pydantic models & enums
│   │   ├── services/       # TRIBE, Claude, pipeline, storage
│   │   ├── utils/          # Video processing helpers
│   │   ├── data/           # Glasser atlas lookup
│   │   ├── config.py       # Settings (pydantic-settings)
│   │   └── main.py         # App factory & lifespan
│   └── tests/              # pytest suite
├── frontend/
│   ├── src/
│   │   ├── components/     # Upload, Processing, Results views
│   │   ├── hooks/          # useAnalysis polling hook
│   │   ├── api/            # HTTP client
│   │   └── types/          # TypeScript interfaces
│   └── public/
├── .env.example
└── PLAN.md                 # Implementation plan
```

## Environment Variables

See [`.env.example`](.env.example) for all configuration options.

| Variable | Required | Description |
|----------|----------|-------------|
| `NM_ANTHROPIC_API_KEY` | For Claude analysis | Anthropic API key |
| `NM_HF_TOKEN` | For TRIBE v2 download | HuggingFace token |
| `NM_USE_MOCK` | No | Enable mock mode (default: false) |

## License

MIT
