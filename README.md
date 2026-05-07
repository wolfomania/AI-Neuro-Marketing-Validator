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
| ML Model | TRIBE v2 (Meta FAIR) + Llama 3.2-3B |
| LLM | Claude API (Anthropic SDK) |
| Brain Atlas | Glasser HCP MMP 1.0 (360 regions) |
| Deployment | Modal.com (serverless GPU) |

## Quick Start (local)

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

Set `NM_USE_MOCK=true` in your `.env` — runs the full pipeline with synthetic brain data, fake Claude report. Good for UI/frontend development.

> **Note:** `NM_USE_MOCK=true` disables Claude even if you have an API key set. There is no "fake TRIBE + real Claude" mode without code changes.

---

## Deploy to Modal.com

Two deployment modes:

| Mode | Modal class | GPU | Brain data | Claude | Cost |
|------|-------------|-----|-----------|--------|------|
| Mock | `WebService` | No | Synthetic | Fake | $0 |
| Real | `GpuService` | A100 | TRIBE v2 | Real | ~$0.50–2/video |

### Prerequisites

- Python 3.11+ and Node.js 18+ installed locally
- [Modal.com](https://modal.com) account (free tier: $30 credits)
- Anthropic API key (for real mode)
- HuggingFace token with access to `facebook/tribev2` and `meta-llama/Llama-3.2-3B` (for real mode)

### Phase 1 — Mock deployment (~10 min, free)

**Step 1 — Install Modal CLI and authenticate**

```bash
pip install modal
modal token new   # opens browser, log in
```

**Step 2 — Build the frontend** (must happen before every deploy)

```bash
cd frontend
npm install
npm run build
cd ..
ls frontend/dist/index.html   # verify this exists
```

**Step 3 — Create Modal secret**

```bash
modal secret create neuromarketing-secrets \
  NM_USE_MOCK="true" \
  NM_ANTHROPIC_API_KEY="" \
  NM_HF_TOKEN="" \
  NM_UPLOAD_DIR="/data/uploads" \
  NM_RESULTS_DIR="/data/results" \
  NM_CORS_ORIGINS="*"
```

To add access token protection (recommended):

```bash
# Generate a token
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Include NM_ACCESS_TOKEN in the secret
modal secret create neuromarketing-secrets \
  NM_USE_MOCK="true" \
  NM_ANTHROPIC_API_KEY="" \
  NM_HF_TOKEN="" \
  NM_UPLOAD_DIR="/data/uploads" \
  NM_RESULTS_DIR="/data/results" \
  NM_CORS_ORIGINS="*" \
  NM_ACCESS_TOKEN="<your-generated-token>"
```

**Step 4 — Smoke test locally**

```bash
modal serve modal_app.py
```

Modal prints a temporary URL. Open it, upload a `.mp4` (15–60s), verify the full UI flow. `Ctrl+C` when done.

**Step 5 — Deploy permanently**

```bash
modal deploy modal_app.py
```

Output:
```
✓ WebService.web  => https://YOUR-WORKSPACE--neuromarketing-webservice-web.modal.run
✓ GpuService.web  => https://YOUR-WORKSPACE--neuromarketing-gpuservice-web.modal.run
```

Use the **`webservice` URL** for mock mode.

### Phase 2 — Real GPU inference

**Step 6 — Download model weights** (run once, ~10 min, billed GPU time)

```bash
modal run modal_app.py::download_model
```

Downloads TRIBE v2 and Llama 3.2-3B into a persistent Modal volume (`nm-model-cache`).

**Step 7 — Update secret with real keys**

```bash
modal secret create neuromarketing-secrets --force \
  NM_USE_MOCK="false" \
  NM_ANTHROPIC_API_KEY="sk-ant-YOUR-KEY" \
  NM_HF_TOKEN="hf_YOUR-TOKEN" \
  NM_UPLOAD_DIR="/data/uploads" \
  NM_RESULTS_DIR="/data/results" \
  NM_CORS_ORIGINS="*" \
  NM_ACCESS_TOKEN="<your-token>"
```

**Step 8 — Redeploy**

```bash
modal deploy modal_app.py
```

Use the **`gpuservice` URL** for real inference.

### Accessing a token-protected deployment

If `NM_ACCESS_TOKEN` is set, the API rejects unauthenticated requests with `401`. The frontend reads the token from `localStorage`. Set it once in the browser console on the deployed site:

```javascript
localStorage.setItem('nm_access_token', 'YOUR-TOKEN-HERE')
```

Refresh. Token persists across sessions. Alternatively, pass it as a query param:

```
https://your-url.modal.run/?token=YOUR-TOKEN-HERE
```

> `/api/health` is always open (no token required).

---

## Useful Commands

```bash
# View logs
modal app logs neuromarketing

# List deployments
modal app list

# Stop a deployment
modal app stop neuromarketing

# Update a secret (--force overwrites)
modal secret create neuromarketing-secrets --force KEY=value ...

# Run model download again
modal run modal_app.py::download_model
```

## Cost Estimates

| Scenario | Monthly cost |
|----------|-------------|
| Mock mode, scale-to-zero | $0 (within free credits) |
| Light GPU usage (~10 videos) | ~$5–20 |
| Always-warm CPU container (24/7) | ~$35 |
| Always-warm A100 GPU (24/7) | ~$2,000+ |

Modal bills per-second. Scale-to-zero (`min_containers=0`) costs nothing while idle. First request after idle takes 30–60s for container spin-up; set `min_containers=1` in `modal_app.py` to avoid cold starts (adds ~$0.05/hr for CPU).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `modal: command not found` | `pip install modal` |
| `Secret not found` | `modal secret list` — check name is `neuromarketing-secrets` |
| Frontend shows blank page | Rebuild: `cd frontend && npm run build`, then redeploy |
| 401 on every request | Set token in browser console: `localStorage.setItem('nm_access_token', '...')` |
| 403 IP not allowed | Check your IP: `curl ifconfig.me`, update `NM_ALLOWED_IPS` in secret |
| Cold start takes 30–60s | Expected. Set `min_containers=1` in `modal_app.py` to skip (costs ~$0.05/hr) |
| Brain images missing | Normal on first run if nilearn fetch fails — pipeline continues without them |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analysis` | Upload video, start analysis |
| `GET` | `/api/analysis/{id}/status` | Poll processing status |
| `GET` | `/api/analysis/{id}` | Get full results with report |
| `GET` | `/api/analysis/{id}/images/{name}` | Brain activation PNGs |
| `GET` | `/api/health` | Health check (always open) |

## Environment Variables

See [`.env.example`](.env.example) for all options.

| Variable | Required | Description |
|----------|----------|-------------|
| `NM_ANTHROPIC_API_KEY` | For Claude analysis | Anthropic API key |
| `NM_HF_TOKEN` | For model download | HuggingFace token (read access) |
| `NM_USE_MOCK` | No | Synthetic mode — skips GPU and Claude (default: `false`) |
| `NM_ACCESS_TOKEN` | No | Bearer token to protect the API |
| `NM_ALLOWED_IPS` | No | Comma-separated IP allowlist (e.g. `1.2.3.4,5.6.7.8`) |
| `NM_TOP_K_ROIS` | No | Brain regions passed to Claude (default: `20`) |
| `NM_CLAUDE_MODEL` | No | Claude model ID (default: `claude-sonnet-4-20250514`) |

## Project Structure

```
.
├── backend/
│   ├── src/neuromarketing/
│   │   ├── api/            # FastAPI routes
│   │   ├── middleware/     # Access guard (IP + token auth)
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
├── modal_app.py            # Modal.com deployment (WebService + GpuService)
└── .env.example
```

## License

MIT
