from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    hf_token: str = ""
    tribe_cache_folder: str = "./cache"
    upload_dir: str = "./uploads"
    results_dir: str = "./results"
    max_video_duration_seconds: int = 60
    min_video_duration_seconds: int = 15
    max_video_size_mb: int = 100
    top_k_rois: int = 20
    claude_model: str = "claude-sonnet-4-20250514"
    cors_origins: str = "http://localhost:5173"
    use_mock: bool = True

    model_config = {"env_file": ".env", "env_prefix": "NM_"}

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def results_path(self) -> Path:
        path = Path(self.results_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_video_size_bytes(self) -> int:
        return self.max_video_size_mb * 1024 * 1024

    def validate_required_keys(self) -> list[str]:
        missing = []
        if not self.anthropic_api_key:
            missing.append("NM_ANTHROPIC_API_KEY")
        if not self.hf_token:
            missing.append("NM_HF_TOKEN")
        return missing
