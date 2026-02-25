from __future__ import annotations

from dataclasses import dataclass
import os



def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}



def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None and raw != "" else default



def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    return raw if raw not in (None, "") else default


@dataclass(frozen=True)
class ApiSettings:
    app_name: str = "Colombian News NLP API"
    app_version: str = "0.1.0"
    app_env: str = "dev"

    sentiment_model_name: str = "ignacio-ave/beto-sentiment-analysis-spanish"
    sentiment_batch_size: int = 16
    sentiment_max_length: int = 512
    sentiment_stride: int = 128
    sentiment_device: str | None = None
    sentiment_num_workers: int = 0
    sentiment_eager_load: bool = False

    classifier_model_path: str | None = None
    classifier_vectorizer_path: str | None = None
    classifier_enabled: bool = True
    classifier_eager_load: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False



def load_settings() -> ApiSettings:
    return ApiSettings(
        app_name=_env_str("API_APP_NAME", "Colombian News NLP API") or "Colombian News NLP API",
        app_version=_env_str("API_APP_VERSION", "0.1.0") or "0.1.0",
        app_env=_env_str("API_ENV", "dev") or "dev",
        sentiment_model_name=_env_str("SENTIMENT_MODEL_NAME", "ignacio-ave/beto-sentiment-analysis-spanish") or "ignacio-ave/beto-sentiment-analysis-spanish",
        sentiment_batch_size=_env_int("SENTIMENT_BATCH_SIZE", 16),
        sentiment_max_length=_env_int("SENTIMENT_MAX_LENGTH", 512),
        sentiment_stride=_env_int("SENTIMENT_STRIDE", 128),
        sentiment_device=_env_str("SENTIMENT_DEVICE", None),
        sentiment_num_workers=_env_int("SENTIMENT_NUM_WORKERS", 0),
        sentiment_eager_load=_env_bool("SENTIMENT_EAGER_LOAD", False),
        classifier_model_path=_env_str("CLASSIFIER_MODEL_PATH", None),
        classifier_vectorizer_path=_env_str("CLASSIFIER_VECTORIZER_PATH", None),
        classifier_enabled=_env_bool("CLASSIFIER_ENABLED", True),
        classifier_eager_load=_env_bool("CLASSIFIER_EAGER_LOAD", False),
        api_host=_env_str("API_HOST", "0.0.0.0") or "0.0.0.0",
        api_port=_env_int("API_PORT", 8000),
        api_reload=_env_bool("API_RELOAD", False),
    )
