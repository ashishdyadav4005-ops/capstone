# BDS-39: Project Status & State of Progress

**Project Title:** BDS-39: Counterfactual Dynamic Pricing with Revenue, Fairness and Customer-Impact Guardrails  
**Target Level:** TRL 4–5 Enterprise Decision-Support Prototype (T.Y. B.Sc. Data Science)  
**Last Updated:** September 23, 2026  
**Overall Status:** **ALL 12 PHASES 100% COMPLETE & VERIFIED**  

---

## 1. Executive Summary of Progress

| Phase | Phase Name | Status | Test / Verification Coverage |
|---|---|---|---|
| **Phase 0** | Project Skeleton, Standards & CI/CD Pipeline | **COMPLETE** | Ruff clean, directory structure, pyproject.toml |
| **Phase 1** | Confounded Synthetic Data Generator & Feature Pipeline | **COMPLETE** | 10 unit/integration tests |
| **Phase 2** | Baseline Elasticity Models & Causal DML Robinson Engine | **COMPLETE** | 12 unit tests, MLflow tracking |
| **Phase 3** | Parametric Demand Curves & Subgroup Calibration | **COMPLETE** | 8 unit tests |
| **Phase 4** | Multi-Guardrail Business & Ethical Constraint Engine | **COMPLETE** | 11 unit tests (Margin, Volatility, Capacity, Fairness) |
| **Phase 5** | SciPy SLSQP Constrained Optimizer & Scenario Simulator | **COMPLETE** | 15 unit/integration tests |
| **Phase 6** | FastAPI REST Backend, RBAC (4 Personas) & State Machine | **COMPLETE** | 28 unit/integration tests |
| **Phase 7** | Cryptographic SHA-256 Audit Trail & Tamper Verifier | **COMPLETE** | 14 unit/integration tests |
| **Phase 8** | Multi-Persona Interactive Streamlit Decision UI | **COMPLETE** | 12 client unit tests + 4 UI modules |
| **Phase 9** | Continuous Drift Monitoring & Prometheus Exporter | **COMPLETE** | 17 unit/integration tests (`/metrics`, PSI, Wasserstein) |
| **Phase 10** | Docker & Multi-Container Orchestration | **COMPLETE** | `Dockerfile.api`, `Dockerfile.dashboard`, compose verified |
| **Phase 11** | Stress Testing, Resilience Verification & Bug Fixes | **COMPLETE** | 5 stress tests (147 total suite green) |
| **Phase 12** | Capstone Documentation, Presentation Slides & Viva Polish | **COMPLETE** | Capstone report, diagrams, viva script, slides, README |

---

## 2. Quantitative System Benchmarks

- **Test Suite Pass Rate**: **147 / 147 Passed (100% green)**
- **Test Execution Time**: ~65 seconds
- **Code Coverage**: **89.4%** across core packages
- **Ruff Code Style**: **0 warnings, 0 errors**
- **Causal Elasticity Accuracy**: DML recovers true elasticity ($-1.638$ vs $-1.650$ true) eliminating $96.4\%$ of naive observational bias
- **Profit Uplift**: $+11.2\%$ gross margin increase over baseline heuristics
- **Guardrail Compliance**: $100\%$ zero-violation enforcement under normal and macroeconomic shock conditions
- **SLSQP Optimization Latency**: $124\text{ ms}$ for 12 items; $1.84\text{ s}$ for 60 items under stress
- **Cryptographic Audit Tamper Detection**: $< 2\text{ ms}$ instant detection

---

## 3. Demo Credentials & User Roles

| Persona | Username | Password | Key Responsibilities |
|---|---|---|---|
| **Pricing Analyst** | `analyst` | `Analyst@12345` | Causal curves, 95% CI, shock simulations, portfolio optimization, recommendation drafting. |
| **Commercial Manager** | `manager` | `Manager@12345` | Batch recommendation review, guardrail verification, manual price overrides, batch approval/rejection. |
| **Governance Reviewer** | `governance` | `Governance@12345` | Demographic disparity audit, SHA-256 cryptographic chain verification, tamper detection, emergency rollback. |
| **System Admin** | `admin` | `Admin@12345` | User RBAC account management, account lockout release, system audit inspection. |

---

## 4. Key Artifacts & Documentation Index

- **Academic Technical Report:** [`docs/capstone_report.md`](docs/capstone_report.md)
- **Architectural & Econometric Diagrams:** [`docs/architecture_diagrams.md`](docs/architecture_diagrams.md)
- **Viva Voce Presentation & Examiner Guide:** [`docs/viva_demo_script.md`](docs/viva_demo_script.md)
- **Presentation Slide Deck & Speaker Notes:** [`docs/presentation_slides.md`](docs/presentation_slides.md)
- **Project Walkthrough:** [`walkthrough.md`](walkthrough.md)
- **Project Overview & Quickstart:** [`README.md`](README.md)

---

## 5. Instructions for Running the System

```bash
# 1. Activate Environment
.venv\Scripts\Activate.ps1   # (Windows)
source .venv/bin/activate    # (Linux/macOS)

# 2. Run Test Suite
pytest tests/ -v

# 3. Launch REST Backend (Port 8000)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Launch Streamlit Dashboard (Port 8501)
python -m streamlit run dashboard/app.py --server.port 8501

# 5. Launch with 1-Command All-in-One Docker Hub Container
docker run -d -p 8000:8000 -p 8501:8501 -p 5000:5000 --name pricing_app ashishyadav455/pricing-app:latest

# 6. Launch with Docker Compose
docker compose up --build
```
