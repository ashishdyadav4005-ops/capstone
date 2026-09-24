"""Pytest fixtures for BDS-39 test suite."""

import os
from pathlib import Path

import pytest

from src.common.config import SystemConfig, load_config


@pytest.fixture
def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def test_config(project_root: Path) -> SystemConfig:
    """Load configuration for testing environment."""
    config_path = project_root / "configs"
    return load_config(config_path)


@pytest.fixture
def clean_env():
    """Fixture to ensure temporary environment variables are restored after tests."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)
