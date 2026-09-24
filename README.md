# BDS-39: Counterfactual Dynamic Pricing with Revenue, Fairness and Customer-Impact Guardrails

[![CI Pipeline](https://github.com/capstone/bds39-pricing/actions/workflows/ci.yml/badge.svg)](https://github.com/capstone/bds39-pricing/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Test Suite](https://img.shields.io/badge/tests-147%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-89.4%25-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TRL 4–5 Enterprise Decision-Support Prototype**  
> An internal decision-support system utilizing **Double Machine Learning (DML)** causal inference and **SciPy constrained optimization (SLSQP)** to recommend profit-maximizing dynamic prices while strictly adhering to business, legal, and ethical guardrails (margin protection, volatility bounds, capacity limits, and demographic/geographic fairness).

---

## 1. System Overview & Core Capabilities

Traditional pricing systems rely on naive observational regressions ($\mathbb{E}[Y \mid P]$), conflating correlation with causation and underestimating price elasticity by upwards of $40\%$ due to seasonal and promotional confounding.

**BDS-39 solves this challenge by integrating:**
1. **Causal Elasticity Estimation**: Robinson’s partially linear Double Machine Learning (DML) with Neyman-orthogonal score equations and 5-fold cross-fitting to isolate true counterfactual elasticities $\frac{\partial \mathbb{E}[Y \mid do(P)]}{\partial P}$ with $95\%$ confidence intervals.
2. **Constrained Portfolio Optimization**: SciPy SLSQP non-linear solver maximizing gross profit $\sum (P_j - C_j) \cdot \min(Q_j(P_j), K_j)$ subject to:
   - **Margin Floor**: Price $\ge$ Unit Cost $\times (1 + 15\%)$
   - **Volatility Envelope**: Price bounded within $\pm 15\%$ of baseline
   - **Capacity Ceiling**: Expected demand $\le 98\%$ of stock
   - **Customer Group Fairness**: Disparity across demographic segments bounded $\le 12\%$
   - **Geographic Fairness**: Disparity across location tiers bounded $\le 10\%$
3. **Cryptographic SHA-256 Audit Ledger**: Forward hash-linked Merkle-style blockchain in SQLite ensuring immutable tamper-evident logging of all recommendations, overrides, and approvals.
4. **Continuous Drift & Prometheus Monitoring**: Population Stability Index (PSI) and Wasserstein Distance monitoring with automated OpenMetrics export on `/metrics`.
5. **Interactive Multi-Persona Dashboard**: 4 distinct Streamlit interfaces tailored for **Pricing Analyst**, **Commercial Manager**, **Governance Reviewer**, and **System Admin**.

---

## 2. Multi-Role User Personas & Demo Credentials

| Role | Username | Password | Primary Interface & Permissions |
|---|---|---|---|
| **Pricing Analyst** | `analyst` | `Analyst@12345` | Causal demand curves, 95% CI, shock simulations, portfolio optimization, recommendation drafting. |
| **Commercial Manager** | `manager` | `Manager@12345` | Recommendation review portal, guardrail verification, manual price overrides, batch approval/rejection. |
| **Governance Reviewer** | `governance` | `Governance@12345` | Demographic disparity audit, SHA-256 cryptographic chain verification, tamper detection, emergency rollback. |
| **System Admin** | `admin` | `Admin@12345` | User RBAC account management, account lockout release, system audit logs. |

---

## 3. Quickstart Guide (Local Setup)

### Prerequisites
- Python 3.11+
- Virtualenv or Conda
- Git

### Step 1: Clone and Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/capstone/bds39-pricing.git
cd capstone_project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Step 2: Seed Database & Sample Data
```bash
# Seed users, synthetic transaction datasets, and baseline models
python scripts/seed_users.py
python scripts/generate_data.py
python scripts/train.py
```

### Step 3: Run the Applications
```bash
# Terminal 1: Launch FastAPI REST backend (Port 8000)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Launch Streamlit multi-role dashboard (Port 8501)
python -m streamlit run dashboard/app.py --server.port 8501
```

Access the interfaces:
- **Streamlit Decision Dashboard:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Interactive API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Prometheus OpenMetrics:** [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

## 4. Docker Deployment

### Option A: 1-Command Run from Docker Hub (All-in-One Single Image)
The entire application stack (FastAPI Backend, Streamlit Decision Dashboard, and MLflow Server) is bundled into a single unified container image on Docker Hub:

```bash
# Pull and run with a single command
docker run -d -p 8000:8000 -p 8501:8501 -p 5000:5000 --name pricing_app ashishyadav455/pricing-app:latest
```

Access the interfaces:
- **Streamlit Decision Dashboard:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Prometheus Metrics:** [http://localhost:8000/metrics](http://localhost:8000/metrics)
- **MLflow Tracking Server:** [http://localhost:5000](http://localhost:5000)

### Option B: Local Multi-Container Compose
Alternatively, launch via Docker Compose:

```bash
docker compose up --build
```

---

## 5. Verification & Testing

The repository contains **147 automated unit, integration, and stress tests** with $100\%$ pass rate.

```bash
# Run all automated tests
pytest tests/ -v

# Run with test coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run code style and lint checks
ruff check .
```

---

## 6. Project Documentation Index

- [Capstone Technical Report](docs/capstone_report.md): In-depth academic report covering econometrics, optimization formulas, and empirical benchmarks.
- [Architecture & Dataflow Diagrams](docs/architecture_diagrams.md): Mermaid diagrams for DML DAG, SLSQP flow, state machine, and cryptographic chain.
- [Viva Demonstration Script](docs/viva_demo_script.md): Examiner walkthrough guide with Q&A cheatsheet.
- [Presentation Slides & Speaker Notes](docs/presentation_slides.md): 15-slide capstone defence presentation outline.
- [Model Card](docs/model_card.md): Causal DML model transparency and limitations documentation.
- [Data Dictionary](docs/data_dictionary.md): Schema specifications for raw and engineered transaction datasets.

---

## 7. Project Structure

```text
capstone_project/
├── configs/                   # YAML configurations (app, guardrails, optimizer)
├── dashboard/                 # Streamlit multi-persona decision-support frontend
│   ├── app.py                 # Main entrypoint with session & auth routing
│   ├── api_client.py          # HTTP client communicating with FastAPI
│   └── views/                 # Dedicated persona UI modules (analyst, manager, governance, admin)
├── docs/                      # Capstone report, diagrams, viva script, slides
├── scripts/                   # CLI scripts (seed_users, generate_data, train, run_optimizer)
├── src/                       # Core package
│   ├── api/                   # FastAPI routers, lifecycle state machine, schemas
│   ├── audit/                 # Cryptographic SHA-256 hash-chained ledger & verifier
│   ├── auth/                  # RBAC, JWT, account lockout security
│   ├── common/                # YAML configuration loader & JSON logging
│   ├── data/                  # Synthetic data generator, validation, feature engineering
│   ├── models/                # Baseline models & Causal DML Robinson residualizer
│   ├── monitoring/            # PSI, Wasserstein drift detector & Prometheus exporter
│   └── pricing/               # Demand curves, SLSQP optimizer, 5 guardrails, scenario simulator
├── tests/                     # 147 unit, integration, and stress tests
├── Dockerfile.api             # Containerfile for FastAPI backend
├── Dockerfile.dashboard       # Containerfile for Streamlit frontend
├── docker-compose.yml         # Compose configuration
├── Makefile                   # Command shortcuts
└── pyproject.toml             # Build and tooling configuration
```

---

## 8. License
This project is licensed under the MIT License.
