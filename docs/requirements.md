# System Requirements & Product Specification

**Project Title**: BDS-39: Counterfactual Dynamic Pricing with Revenue, Fairness and Customer-Impact Guardrails  
**Target Readiness Level**: TRL 4–5 (Industry-style, end-to-end working prototype)  
**System Classification**: Internal Decision-Support System (Staff only; never customer-facing)

---

## 1. System Vision & Objectives
The BDS-39 system enables revenue managers and data analysts to optimize dynamic prices using **causal machine learning (Double Machine Learning)** to estimate true price elasticity free of confounding bias. The pricing engine calculates optimal revenue subject to strict, auditable **guardrails** for capacity, price volatility, fairness across customer groups and locations, and profit margins.

---

## 2. User Personas & Permissions Matrix (RBAC)

| Role | Primary Persona Description | Core Permissions & System Capabilities |
|---|---|---|
| **Admin** | System & Security Administrator | Create/disable users, assign RBAC roles, manage system configuration, monitor uptime. |
| **Analyst** | Pricing & Data Science Analyst | View causal demand curves, revenue curves, elasticity intervals, run what-if price simulations. Read-only on recommendations. |
| **Manager** | Commercial / Revenue Manager | Everything an Analyst can do, PLUS authority to **Approve, Reject, or Override** prices (reason mandatory). |
| **Governance Reviewer** | Compliance, Legal & Ethics Officer | Read-only oversight: fairness metrics, subgroup impact reports, constraint violations, tamper-evident audit logs, model health/drift. |

---

## 3. Misuse, Abuse, and Threat Scenarios

| Threat / Abuse Case | Actor | Vector / Mechanism | Mitigation Strategy |
|---|---|---|---|
| **Privilege Escalation** | Analyst / External | Analyst attempts to approve or override pricing recommendations via direct API call. | Strict RBAC dependency on backend routes with JWT role verification. Server-side default deny. |
| **Unauthorized Price Manipulation** | Malicious / Compromised User | Injecting arbitrary pricing multipliers to cause sudden drastic price spikes. | Volatility guardrail ($\le \pm X\%$), margin floor, mandatory manager approval, tamper-evident audit logging. |
| **Data Poisoning** | Malicious Insider | Injecting corrupted demand/price rows into data pipeline to distort elasticity. | Automated data validation schemas (pydantic/pandera), distribution outlier detection, drift alerts. |
| **Credential Stuffing & Brute Force** | External Attacker | Rapid automated login attempts against the auth endpoint. | Account lockout after 5 failed attempts, login rate limiting, bcrypt salted hashing. |
| **Audit Trail Tampering** | Rogue Admin | Modifying database records to hide unauthorized price overrides. | Cryptographic SHA-256 hash chaining of audit entries; integrity verification routine. |

---

## 4. Scope Exclusions (What the System is NOT)
- **Not Customer-Facing**: The system does not serve live consumer traffic; it outputs approved prices to downstream catalog/e-commerce feeds.
- **Not a Black-Box Auto-Pricer**: Fully automated live bidding without human-in-the-loop review is out of scope; managers review and approve recommendations.
- **Not a Generic BI Dashboard**: It is a domain-specific causal decision-support tool, not an ad-hoc reporting portal.

---

## 5. Success Metrics & Quality Gates

### Quantitative Metrics
- **Out-of-Sample Demand Error**: Test set $\text{MAPE} \le 12\%$, $\text{RMSE}$ competitive with naive baseline.
- **Causal Elasticity Accuracy**: Estimated elasticity $\hat{\varepsilon}$ within $\pm 10\%$ of known simulated ground truth $\varepsilon^*$.
- **Revenue Uplift**: Expected revenue gain $\ge 5\%$ over static/heuristic cost-plus pricing.
- **Guardrail Compliance**: $0\%$ unapproved hard-constraint violations (capacity, margin floor, volatility).
- **Fairness Disparity**: Price disparity between geographic locations/demographic groups bounded by $\le 10\%$.
- **System Performance**: Optimization & simulation runtime $< 500\text{ms}$ per product.

---

## 6. Prioritized Product Backlog

| Phase | Priority | Module | Deliverable |
|---|---|---|---|
| **Phase 0** | P0 | Foundation | Repo skeleton, configs, logging, Makefile, Docker, CI, docs. |
| **Phase 1** | P0 | Data Layer | Synthetic transaction generator with confounding, validation, feature engineering, leakage checks. |
| **Phase 2** | P0 | Modeling | Double Machine Learning (DML) causal elasticity, baseline models, sensitivity analysis, MLflow tracking. |
| **Phase 3** | P0 | Demand Curves | Counterfactual demand curve generator with confidence intervals, cold-start handling. |
| **Phase 4** | P0 | Optimizer | SciPy constrained revenue optimization under capacity, volatility, margin, and fairness guardrails. |
| **Phase 5** | P0 | Backend API | FastAPI RESTful service, state machine (`GENERATED` $\rightarrow$ `PUBLISHED`), OpenAPI schema. |
| **Phase 6** | P0 | Auth & RBAC | JWT authentication, role enforcement, lockout policy, seed script. |
| **Phase 7** | P1 | Audit Trail | Tamper-evident hash-chained audit log, policy compliance audit engine. |
| **Phase 8** | P0 | Dashboard | Streamlit multi-role interface with what-if simulator and decision panels. |
| **Phase 9** | P1 | Monitoring | Prometheus metrics, data/concept drift detection (PSI/KS), model health tracking. |
| **Phase 10** | P0 | Evaluation | Comprehensive evaluation dossier, baseline comparison tables, stress testing. |
| **Phase 11** | P1 | Hardening | Test coverage $\ge 80\%$, security audit (`pip-audit`), container orchestration. |
| **Phase 12** | P1 | Documentation | C4 architecture diagrams, model cards, demo script, contribution evidence. |
