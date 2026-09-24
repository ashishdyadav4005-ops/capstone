"""Configuration management module supporting YAML files and environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "BDS-39 Dynamic Pricing"
    version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class AuthConfig(BaseModel):
    jwt_secret_key: str = "dev-secret-key-please-change-in-production-12345678"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    max_failed_logins: int = 5
    lockout_duration_minutes: int = 15


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///./data/bds39_pricing.db"


class MLflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///./data/mlflow.db"
    experiment_name: str = "bds39_counterfactual_pricing"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"


class SystemConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    raw_configs: dict[str, Any] = Field(default_factory=dict)


_GLOBAL_CONFIG: SystemConfig | None = None


def find_project_root() -> Path:
    """Locate the project root directory by searching for markers like pyproject.toml or configs/."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / "configs").exists():
            return parent
    return Path.cwd()


def load_yaml_file(file_path: Path) -> dict[str, Any]:
    """Safely load and parse a YAML file."""
    if not file_path.exists():
        return {}
    with open(file_path, encoding="utf-8") as f:
        content = yaml.safe_load(f)
        return content or {}


def load_config(config_dir: Path | str | None = None) -> SystemConfig:
    """Load system configuration from YAML files and environment variables."""
    global _GLOBAL_CONFIG

    root_path = find_project_root()
    base_dir = Path(config_dir) if config_dir else root_path / "configs"

    merged: dict[str, Any] = {}

    # Load all YAML config files in configs directory
    yaml_files = ["app.yaml", "data.yaml", "model.yaml", "optimizer.yaml", "guardrails.yaml"]
    for yml in yaml_files:
        path = base_dir / yml
        if path.exists():
            data = load_yaml_file(path)
            # Store in top-level or raw_configs
            for k, v in data.items():
                merged[k] = v

    # Apply environment variable overrides
    app_data = merged.get("app", {})
    if "ENVIRONMENT" in os.environ:
        app_data["environment"] = os.environ["ENVIRONMENT"]

    auth_data = merged.get("auth", {})
    if "JWT_SECRET_KEY" in os.environ:
        auth_data["jwt_secret_key"] = os.environ["JWT_SECRET_KEY"]
    if "JWT_ALGORITHM" in os.environ:
        auth_data["jwt_algorithm"] = os.environ["JWT_ALGORITHM"]
    if "ACCESS_TOKEN_EXPIRE_MINUTES" in os.environ:
        auth_data["access_token_expire_minutes"] = int(os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"])

    db_data = merged.get("database", {})
    if "DATABASE_URL" in os.environ:
        db_data["url"] = os.environ["DATABASE_URL"]

    mlflow_data = merged.get("mlflow", {})
    if "MLFLOW_TRACKING_URI" in os.environ:
        mlflow_data["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    if "MLFLOW_EXPERIMENT_NAME" in os.environ:
        mlflow_data["experiment_name"] = os.environ["MLFLOW_EXPERIMENT_NAME"]

    log_data = merged.get("logging", {})
    if "LOG_LEVEL" in os.environ:
        log_data["level"] = os.environ["LOG_LEVEL"]

    server_data = merged.get("server", {})
    if "API_HOST" in os.environ:
        server_data["host"] = os.environ["API_HOST"]
    if "API_PORT" in os.environ:
        server_data["port"] = int(os.environ["API_PORT"])

    system_config = SystemConfig(
        app=AppConfig(**app_data) if app_data else AppConfig(),
        server=ServerConfig(**server_data) if server_data else ServerConfig(),
        auth=AuthConfig(**auth_data) if auth_data else AuthConfig(),
        database=DatabaseConfig(**db_data) if db_data else DatabaseConfig(),
        mlflow=MLflowConfig(**mlflow_data) if mlflow_data else MLflowConfig(),
        logging=LoggingConfig(**log_data) if log_data else LoggingConfig(),
        raw_configs=merged
    )

    _GLOBAL_CONFIG = system_config
    return system_config


def get_config() -> SystemConfig:
    """Get the active global system configuration (or load default if not yet loaded)."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = load_config()
    return _GLOBAL_CONFIG
