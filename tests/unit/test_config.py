"""Unit tests for configuration loader and environment overrides."""

import os
from pathlib import Path

from src.common.config import SystemConfig, find_project_root, get_config, load_config


def test_find_project_root():
    root = find_project_root()
    assert isinstance(root, Path)
    assert (root / "configs").exists() or (root / "pyproject.toml").exists()


def test_load_default_config(test_config: SystemConfig):
    assert test_config is not None
    assert test_config.app.name == "BDS-39 Counterfactual Dynamic Pricing System"
    assert test_config.server.port == 8000
    assert test_config.database.url.startswith("sqlite")
    assert "simulation" in test_config.raw_configs


def test_env_overrides(clean_env):
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["JWT_SECRET_KEY"] = "super-secret-test-token"
    os.environ["API_PORT"] = "9000"

    config = load_config()
    assert config.app.environment == "testing"
    assert config.logging.level == "DEBUG"
    assert config.auth.jwt_secret_key == "super-secret-test-token"
    assert config.server.port == 9000


def test_get_config_singleton():
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2
