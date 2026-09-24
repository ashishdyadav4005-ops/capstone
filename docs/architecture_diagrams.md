# BDS-39: System Architecture & Dataflow Diagrams

This document collects the complete set of architectural, econometric, mathematical, security, and lifecycle diagrams for **BDS-39: Counterfactual Dynamic Pricing with Revenue, Fairness and Customer-Impact Guardrails**.

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph ClientLayer["User Interface Layer (Streamlit)"]
        UI_Analyst["Analyst View\n(Simulation & Elasticity)"]
        UI_Manager["Manager View\n(Approval & Overrides)"]
        UI_Gov["Governance View\n(Audit & Drift Monitor)"]
        UI_Admin["Admin View\n(User & Security Config)"]
    end

    subgraph APILayer["FastAPI REST Backend (Port 8000)"]
        AuthRouter["/api/v1/auth\n(JWT & RBAC Gate)"]
        PricingRouter["/api/v1/pricing\n(Curves, SLSQP, Lifecycle)"]
        AuditRouter["/api/v1/audit\n(Hash Verifier & Ledger)"]
        DriftRouter["/api/v1/monitoring\n(PSI, KS, Prometheus /metrics)"]
    end

    subgraph CoreEngine["Core Algorithmic Engines"]
        DML["Double Machine Learning (DML)\n(Robinson Residualizer & 5-Fold)"]
        Optimizer["Constrained Price Optimizer\n(SciPy SLSQP & 5 Guardrails)"]
        AuditLedger["Cryptographic Audit Chain\n(SHA-256 Merkle Ledger)"]
        DriftEngine["Drift & Metric Engine\n(PSI, Wasserstein, KS-Test)"]
    end

    subgraph StorageLayer["Persistence & Artifacts"]
        SQLiteDB[("SQLite Storage\n(Users, Recs, Audit Logs)")]
        ParquetData[("Parquet Lake\n(Transactions & Features)")]
        MLflowStore[("MLflow Tracking\n(Model Artifacts & Runs)")]
    end

    UI_Analyst -->|HTTP / Bearer JWT| APILayer
    UI_Manager -->|HTTP / Bearer JWT| APILayer
    UI_Gov -->|HTTP / Bearer JWT| APILayer
    UI_Admin -->|HTTP / Bearer JWT| APILayer

    APILayer --> CoreEngine
    CoreEngine --> StorageLayer
```

---

## 2. Causal Elasticity Estimation (Double Machine Learning DAG)

```mermaid
flowchart LR
    subgraph Confounders["Confounding Covariates (W)"]
        C1["Competitor Pricing"]
        C2["Seasonality & Day-of-Week"]
        C3["Inventory Days-of-Supply"]
        C4["Regional Tier & Demographics"]
    end

    subgraph Models["Nuisance Machine Learning Estimators"]
        NuisanceG["Outcome Model: g(W) = E[Demand | W]\n(LightGBM / GBRT)"]
        NuisanceM["Propensity Model: m(W) = E[Price | W]\n(LightGBM / GBRT)"]
    end

    subgraph Residuals["Orthogonal Residuals (5-Fold Cross-Fitted)"]
        ResY["Y_tilde = Demand - g(W)"]
        ResP["P_tilde = Price - m(W)"]
    end

    subgraph CausalParameter["Causal Effect Estimation"]
        ThetaEstimator["theta = sum(P_tilde * Y_tilde) / sum(P_tilde^2)\n+ 95% Confidence Interval"]
    end

    Confounders --> NuisanceG
    Confounders --> NuisanceM
    NuisanceG --> ResY
    NuisanceM --> ResP
    ResY --> ThetaEstimator
    ResP --> ThetaEstimator
```

---

## 3. Constrained SLSQP Non-Linear Portfolio Optimization Flow

```mermaid
flowchart TD
    Init["1. Input Baseline Prices P0, Costs C, Capacities K"] --> BuildObjective["2. Formulate Profit Objective: max sum(P - C) * min(Q(P), K)"]
    BuildObjective --> SetBounds["3. Establish Volatility Box Bounds: [0.85 * P0, 1.15 * P0]"]
    SetBounds --> AttachConstraints["4. Attach Hard Non-Linear Inequality Constraints"]

    subgraph Guardrails["Multi-Guardrail Verification Engine"]
        G1["Margin Floor: P >= Cost * (1 + 0.15)"]
        G2["Capacity Limit: Q(P) <= 0.98 * Capacity"]
        G3["Group Fairness: max(P_g) - min(P_g) <= 0.12 * min(P_g)"]
        G4["Geo Fairness: max(P_loc) - min(P_loc) <= 0.10 * min(P_loc)"]
    end

    AttachConstraints --> Guardrails
    Guardrails --> SLSQPSolver["5. Execute SciPy SLSQP Solver\n(Han-Powell Quasi-Newton Hessian)"]
    SLSQPSolver --> CheckFeasibility{"Feasible & Converged?"}
    CheckFeasibility -- Yes --> OutputOptimal["Output Optimal Compliant Prices P*"]
    CheckFeasibility -- No / Fallback --> SafeFallback["Fallback: Clip to Safe Volatility & Margin Bounds"]
```

---

## 4. Recommendation Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> DRAFT : Analyst triggers SLSQP optimizer
    DRAFT --> SIMULATED : Analyst runs stress scenario shock
    SIMULATED --> DRAFT : Parameter adjustment
    DRAFT --> SUBMITTED : Analyst submits recommendation batch
    SIMULATED --> SUBMITTED : Analyst submits validated scenario
    
    SUBMITTED --> APPROVED : Manager approves batch
    SUBMITTED --> REJECTED : Manager rejects (Reason required)
    SUBMITTED --> APPLIED : Direct auto-application (if pre-authorized)
    
    APPROVED --> APPLIED : System dispatches prices to POS/Web
    APPLIED --> ROLLED_BACK : Governance or System emergency rollback
    
    REJECTED --> [*]
    ROLLED_BACK --> [*]
```

---

## 5. Cryptographic SHA-256 Hash Chain Structure

```mermaid
flowchart LR
    subgraph Genesis["Genesis Block (ID: 0)"]
        G_Payload["ID: 0\nEvent: GENESIS\nActor: system\nDetails: Chain initialized"]
        G_Prev["Prev Hash: 0000000000000000..."]
        G_Hash["Current Hash: a3f89e21..."]
    end

    subgraph Block1["Audit Block 1 (ID: 1)"]
        B1_Payload["ID: 1\nEvent: OPTIMIZATION_RUN\nActor: analyst_bob\nDetails: Portfolio P01-P12 solved"]
        B1_Prev["Prev Hash: a3f89e21..."]
        B1_Hash["Current Hash: 7c4e11a0..."]
    end

    subgraph Block2["Audit Block 2 (ID: 2)"]
        B2_Payload["ID: 2\nEvent: RECOMMENDATION_APPROVED\nActor: manager_alice\nDetails: Approved Batch #104"]
        B2_Prev["Prev Hash: 7c4e11a0..."]
        B2_Hash["Current Hash: e9b433cf..."]
    end

    Genesis -->|Previous Hash Link| Block1
    Block1 -->|Previous Hash Link| Block2
```

---

## 6. Continuous Drift & Prometheus Metric Pipeline

```mermaid
flowchart LR
    subgraph LiveTraffic["Live Ingestion Window"]
        Tx["Streaming Transactions\n(Price, Sales, Costs)"]
    end

    subgraph DriftEngine["Statistical Drift Detector"]
        PSI["Population Stability Index (PSI)\n(10 Quantile Bins)"]
        Wasserstein["Wasserstein-1 Distance\n(Continuous EMD)"]
        KS["2-Sample Kolmogorov-Smirnov\n(P-value & Max Sup)"]
    end

    subgraph MetricsExport["Prometheus OpenMetrics Exporter"]
        PromMetrics["/metrics Endpoint\n- pricing_drift_psi\n- pricing_drift_wasserstein\n- pricing_optimization_duration_seconds\n- pricing_guardrail_violations_total"]
    end

    subgraph Alerts["Observability & Alerting"]
        PrometheusServer["Prometheus / Grafana"]
        AlertManager["AlertManager: PSI >= 0.25 (Retrain Trigger)"]
    end

    LiveTraffic --> DriftEngine
    DriftEngine --> MetricsExport
    MetricsExport --> PrometheusServer
    PrometheusServer --> AlertManager
```
