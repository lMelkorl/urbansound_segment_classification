"""Static model readiness inspection without runtime model imports."""

from .preflight import MODEL_PREFLIGHT_SCHEMA_VERSION, run_model_preflight

__all__ = ["MODEL_PREFLIGHT_SCHEMA_VERSION", "run_model_preflight"]
