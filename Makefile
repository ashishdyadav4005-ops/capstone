.PHONY: help setup data train optimize api dashboard test lint format clean docker-build docker-up docker-down

PYTHON ?= python

help:
	@echo "BDS-39 Counterfactual Dynamic Pricing System"
	@echo "Available commands:"
	@echo "  make setup      - Install package in editable mode with dev dependencies"
	@echo "  make data       - Generate simulated transactions dataset"
	@echo "  make train      - Train baseline and causal elasticity models"
	@echo "  make optimize   - Run pricing optimization engine"
	@echo "  make seed-users - Seed initial demo users for RBAC roles"
	@echo "  make api        - Start FastAPI backend server"
	@echo "  make dashboard  - Start Streamlit frontend app"
	@echo "  make test       - Run unit and integration tests with pytest"
	@echo "  make lint       - Run ruff linter and code formatting check"
	@echo "  make format     - Automatically format code with ruff and black"
	@echo "  make clean      - Clean temporary files and caches"
	@echo "  make docker-up  - Launch full system with docker-compose"
	@echo "  make docker-down- Stop all docker services"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) scripts/generate_data.py

train:
	$(PYTHON) scripts/train.py

optimize:
	$(PYTHON) scripts/run_optimizer.py

seed-users:
	$(PYTHON) scripts/seed_users.py

api:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/ dashboard/
	$(PYTHON) -m black --check src/ tests/ scripts/ dashboard/

format:
	$(PYTHON) -m ruff check --fix src/ tests/ scripts/ dashboard/
	$(PYTHON) -m black src/ tests/ scripts/ dashboard/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
