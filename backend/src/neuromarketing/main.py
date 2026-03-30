"""FastAPI application entry point."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neuromarketing.api.router import api_router
from neuromarketing.config import Settings
from neuromarketing.middleware.access_guard import AccessGuardMiddleware, build_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Validate environment, create directories, and initialise services."""
    settings = Settings()

    # Warn about missing keys (don't crash -- allows dev without keys)
    missing = settings.validate_required_keys()
    if missing:
        logger.warning(
            "Missing environment variables (needed for full functionality): %s",
            ", ".join(missing),
        )

    # Ensure upload and results directories exist
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.results_path.mkdir(parents=True, exist_ok=True)

    logger.info("Upload dir:  %s", settings.upload_path.resolve())
    logger.info("Results dir: %s", settings.results_path.resolve())

    # Initialise TRIBE service
    if settings.use_mock:
        from neuromarketing.services.mock_tribe import MockTribeService

        tribe_service = MockTribeService(top_k=settings.top_k_rois)
        tribe_service.load_model()
        logger.info("Using MockTribeService (NM_USE_MOCK=true)")
    else:
        from neuromarketing.services.tribe_service import TribeService

        tribe_service = TribeService(
            cache_folder=settings.tribe_cache_folder,
            hf_token=settings.hf_token,
            top_k=settings.top_k_rois,
        )
        tribe_service.load_model()
        logger.info("Using real TribeService with GPU")

    # Initialise Claude client (None in mock mode without API key)
    claude_client = None
    if settings.anthropic_api_key and not settings.use_mock:
        try:
            import anthropic

            claude_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            logger.info("Anthropic client initialised")
        except ImportError:
            logger.warning(
                "anthropic package not installed; Claude analysis disabled"
            )
    else:
        logger.info("Claude client disabled (mock mode or no API key)")

    # Store services on app.state for dependency injection
    app.state.tribe_service = tribe_service
    app.state.claude_client = claude_client
    app.state.settings = settings

    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = Settings()

    app = FastAPI(
        title="AI Neuro-Marketing Validator",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS -- origins from comma-separated config value
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Access control -- enabled via NM_ACCESS_TOKEN and/or NM_ALLOWED_IPS
    guard_config = build_config(
        access_token=settings.access_token,
        allowed_ips_raw=settings.allowed_ips,
    )
    if guard_config.is_enabled:
        app.add_middleware(AccessGuardMiddleware, config=guard_config)
        logger.info("Access guard enabled (token=%s, ips=%d)",
                     bool(guard_config.access_token), len(guard_config.allowed_ips))

    app.include_router(api_router)

    return app


app = create_app()
