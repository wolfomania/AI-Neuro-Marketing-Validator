"""Modal.com deployment for AI Neuro-Marketing Validator.

Usage:
    modal serve modal_app.py          # dev mode (temporary URL, hot-reload)
    modal deploy modal_app.py         # production (persistent URL)
    modal run modal_app.py::download_model   # one-time model weight download
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("neuromarketing")

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
upload_vol = modal.Volume.from_name("nm-uploads", create_if_missing=True)
results_vol = modal.Volume.from_name("nm-results", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("nm-model-cache", create_if_missing=True)

VOLUMES = {
    "/data/uploads": upload_vol,
    "/data/results": results_vol,
    "/data/model-cache": model_cache_vol,
}

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
frontend_dist = Path(__file__).parent / "frontend" / "dist"
backend_src = Path(__file__).parent / "backend" / "src"

_base_deps = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi[standard]>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "python-multipart>=0.0.9",
        "pydantic-settings>=2.0.0",
        "anthropic>=0.40.0",
        "ffmpeg-python>=0.2.0",
        "numpy>=1.24.0",
        "httpx>=0.27.0",
    )
)

# add_local_* must come last — no build steps allowed after it
image = (
    _base_deps
    .add_local_dir(str(backend_src), remote_path="/app/src")
    .add_local_dir(str(frontend_dist), remote_path="/assets")
)

# GPU base: install git (needed for tribev2 git URL), then pip deps,
# then pre-download spacy model so cold starts don't pay 400MB download.
_gpu_deps = (
    _base_deps
    .apt_install("git")
    .pip_install(
        "torch>=2.5.0,<2.7.0",
        "huggingface-hub>=0.20.0",
        "transformers>=4.40.0",
        "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git",
    )
    .run_commands(
        "python -m spacy download en_core_web_lg",
        # Pre-cache MNE sample data + HCP atlas so cold starts skip 1.6GB download.
        (
            "python -c '"
            "import mne; "
            "p = str(mne.datasets.sample.data_path()) + \"/subjects\"; "
            "mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=p, accept=True)"
            "'"
        ),
    )
)

gpu_image = (
    _gpu_deps
    .add_local_dir(str(backend_src), remote_path="/app/src")
    .add_local_dir(str(frontend_dist), remote_path="/assets")
)

MINUTES = 60

# ---------------------------------------------------------------------------
# Main web service (mock mode — no GPU required)
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("neuromarketing-secrets")],
    volumes=VOLUMES,
    scaledown_window=5 * MINUTES,
    min_containers=0,
    timeout=10 * MINUTES,
)
class WebService:
    """Serves React frontend + FastAPI backend in mock mode (no GPU)."""

    @modal.enter()
    def startup(self) -> None:
        os.environ.setdefault("NM_UPLOAD_DIR", "/data/uploads")
        os.environ.setdefault("NM_RESULTS_DIR", "/data/results")
        os.environ.setdefault("NM_TRIBE_CACHE_FOLDER", "/data/model-cache")
        import sys

        sys.path.insert(0, "/app/src")

    @modal.asgi_app()
    def web(self) -> object:
        import sys

        import fastapi.staticfiles

        sys.path.insert(0, "/app/src")
        from neuromarketing.main import create_app

        web_app = create_app()
        web_app.mount(
            "/",
            fastapi.staticfiles.StaticFiles(directory="/assets", html=True),
        )
        return web_app


# ---------------------------------------------------------------------------
# GPU service (production — real TRIBE v2 inference)
# ---------------------------------------------------------------------------


@app.cls(
    image=gpu_image,
    gpu="A100",
    secrets=[modal.Secret.from_name("neuromarketing-secrets")],
    volumes=VOLUMES,
    scaledown_window=5 * MINUTES,
    min_containers=0,
    timeout=30 * MINUTES,
)
class GpuService:
    """Serves React frontend + FastAPI backend with real GPU inference."""

    @modal.enter()
    def startup(self) -> None:
        os.environ.setdefault("NM_UPLOAD_DIR", "/data/uploads")
        os.environ.setdefault("NM_RESULTS_DIR", "/data/results")
        os.environ.setdefault("NM_TRIBE_CACHE_FOLDER", "/data/model-cache")
        os.environ.setdefault("NM_USE_MOCK", "false")
        import sys

        sys.path.insert(0, "/app/src")

    @modal.asgi_app()
    def web(self) -> object:
        import sys

        import fastapi.staticfiles

        sys.path.insert(0, "/app/src")
        from neuromarketing.main import create_app

        web_app = create_app()
        web_app.mount(
            "/",
            fastapi.staticfiles.StaticFiles(directory="/assets", html=True),
        )
        return web_app


# ---------------------------------------------------------------------------
# One-time model download
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    secrets=[modal.Secret.from_name("neuromarketing-secrets")],
    volumes={"/data/model-cache": model_cache_vol},
    timeout=30 * MINUTES,
)
def download_model() -> None:
    """Pre-download TRIBE v2 weights to persistent volume.

    Run once:  modal run modal_app.py::download_model
    """
    os.environ["HF_HOME"] = "/data/model-cache"
    token = os.environ.get("NM_HF_TOKEN")
    from huggingface_hub import snapshot_download

    for repo in ("facebook/tribev2", "meta-llama/Llama-3.2-3B"):
        logger.info("Downloading %s ...", repo)
        snapshot_download(repo, cache_dir="/data/model-cache", token=token)

    model_cache_vol.commit()
    logger.info("Model weights committed to volume.")
