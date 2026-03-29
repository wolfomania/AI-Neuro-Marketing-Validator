"""Shared test fixtures."""

from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from neuromarketing.config import Settings
from neuromarketing.services.mock_tribe import MockTribeService


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings with temporary directories for isolation."""
    return Settings(
        anthropic_api_key="test-key",
        hf_token="test-token",
        upload_dir=str(tmp_path / "uploads"),
        results_dir=str(tmp_path / "results"),
        cors_origins="http://localhost:3000",
    )


@pytest.fixture
async def client(test_settings: Settings) -> AsyncClient:
    """Async test client with patched settings."""
    from neuromarketing.api.analysis import get_settings
    from neuromarketing.main import create_app

    app = create_app()

    # Override the settings dependency
    app.dependency_overrides[get_settings] = lambda: test_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def mock_tribe_service() -> MockTribeService:
    """Provide a loaded MockTribeService for use across test modules."""
    svc = MockTribeService(top_k=20)
    svc.load_model()
    return svc
