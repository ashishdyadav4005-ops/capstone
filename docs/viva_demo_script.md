# BDS-39: Viva Voce & Capstone Demonstration Guide

## 1. Demonstration Setup & Credentials

### 1.1 Local Launch Commands
```bash
# Terminal 1: Launch FastAPI Backend (Port 8000)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Launch Multi-Persona Streamlit Dashboard (Port 8501)
python -m streamlit run dashboard/app.py --server.port 8501
```

### 1.2 Multi-Persona Demo Credentials
| Persona | Username | Password | Key Capabilities to Demonstrate |
|---|---|---|---|
| **Pricing Analyst** | `analyst` | `Analyst@12345` | Causal demand curves, 95% CI, shock simulation, portfolio optimization, recommendation drafting. |
| **Commercial Manager** | `manager` | `Manager@12345` | Batch recommendation review, margin/guardrail verification, manual price override, approval/rejection. |
| **Governance Reviewer** | `governance` | `Governance@12345` | Demographic disparity audit, SHA-256 cryptographic chain verification, tamper attack detection, emergency rollback. |
| **System Admin** | `admin` | `Admin@12345` | User RBAC account management, failed login lockout release, system audit inspection. |

---

## 2. Step-by-Step Viva Demonstration Walkthrough

### Flow 1: Pricing Analyst — Causal Inference & What-If Simulation
1. **Login**: Authenticate as `analyst` (`Analyst@12345`).
2. **Examine Demand Curve**:
   - Navigate to the **Demand Curves & Elasticity** tab.
   - Select `Product: PROD_001` (Premium Wireless Headphones).
   - Point out the **True Causal Elasticity** ($\theta = -1.64$) with finite-sample $95\%$ confidence bounds.
   - Contrast this with the naive OLS curve ($\theta = -0.94$), explaining to the examiner that naive regressions underestimate elasticity due to peak-demand holiday promotions.
3. **What-If Macroeconomic Shock Simulation**:
   - Navigate to the **Scenario Simulator** tab.
   - Select Scenario: **"Hyperinflation / Cost Surge"** (+20% wholesale unit cost).
   - Observe the updated optimal price recommendation and verify that the guardrail engine automatically clamps prices to preserve the $\ge 15\%$ profit margin.
4. **Portfolio Optimization**:
   - Navigate to **Portfolio Optimization**.
   - Trigger the **Joint SLSQP Optimizer** across 12 product-segment pairs.
   - Highlight the sub-second execution latency ($< 150\text{ ms}$) and compliance across all 5 guardrails.
   - Click **"Submit Recommendations for Manager Approval"**.

---

### Flow 2: Commercial Manager — Review, Override & Approval
1. **Login**: Log out and log in as `manager` (`Manager@12345`).
2. **Review Pending Batches**:
   - Navigate to the **Manager Approval Portal**.
   - Inspect the pending recommendations submitted by the Analyst.
3. **Test Manager Price Override**:
   - Select a product and attempt to override the recommended price below cost (e.g., \$35 when unit cost is \$40).
   - Observe the immediate **Margin Floor Guardrail Violation** banner rejecting the override.
   - Correct the override to a valid commercial price (\$58.00) with a mandatory business justification reason: *"Strategic spring bundle alignment"*.
4. **Approve Batch**:
   - Click **"Approve & Dispatch Batch"**.
   - Note the state machine transition from `SUBMITTED` $\to$ `APPROVED` $\to$ `APPLIED`.

---

### Flow 3: Governance Reviewer — Fairness & Cryptographic Audit Verification
1. **Login**: Log out and log in as `governance` (`Governance@12345`).
2. **Disparity & Demographic Fairness Audit**:
   - Navigate to the **Fairness & Disparity Report** tab.
   - Inspect the customer group price disparity chart ($\le 12\%$ hard constraint) and geographic regional disparity ($\le 10\%$).
   - Verify that all customer groups (Budget, Regular, Premium, Student) are protected from predatory price gouging.
3. **Cryptographic SHA-256 Ledger Verification**:
   - Navigate to the **Cryptographic Audit Ledger** tab.
   - Click **"Run Full Audit Chain Verification"**.
   - Observe the **100% Chain Integrity Verified** banner confirming all forward SHA-256 hash pointers link back to the genesis block.
4. **Live Tamper Defense Demonstration (Examiner Highlight)**:
   - Explain how the system prevents database tampering: if a malicious database administrator directly edits SQLite to manipulate an approval record, the next verification run flags the exact corrupted block and halts price dispatch.
5. **Continuous Drift & Prometheus Monitoring**:
   - View the **Drift Dashboard** showing real-time Population Stability Index (PSI) and Wasserstein Distance.
   - Inspect Prometheus `/metrics` endpoint showing real-time counters and histograms.

---

## 3. High-Yield Viva Voce Technical Q&A

### Q1: Why did you choose Double Machine Learning (DML) over standard regression or neural networks?
> **Answer:** Standard regression and deep neural networks are predictive models designed for $\mathbb{E}[Y \mid P, W]$. They do not identify the causal estimand $\frac{\partial \mathbb{E}[Y \mid do(P)]}{\partial P}$. Because prices are historically raised during high-demand periods, observational data is plagued by endogeneity ($\text{Cov}(P, \varepsilon) \neq 0$). DML uses Robinson's residualization with Neyman-orthogonal score functions and 5-fold cross-fitting to achieve $\sqrt{N}$-consistent, asymptotically unbiased elasticity estimation without regularizer bias.

### Q2: How does the optimizer ensure customer fairness across demographic groups?
> **Answer:** We formulate demographic fairness as a coupled non-linear inequality constraint in SciPy SLSQP:
> $$\frac{\max_{g} P_{j, g} - \min_{g} P_{j, g}}{\min_{g} P_{j, g}} \le 0.12$$
> This prevents the algorithm from charging vulnerable segments more than a $12\%$ disparity relative to baseline segments for identical goods, satisfying both corporate ethical charters and emerging AI pricing regulations.

### Q3: What is the purpose of the SHA-256 hash-chained audit trail?
> **Answer:** In high-stakes algorithmic pricing, accountability and non-repudiation are mandatory. By hashing each event's canonical JSON payload alongside the previous block's SHA-256 hash, we construct an immutable Merkle-style blockchain ledger within SQLite. Any retroactive alteration of prices, timestamps, or manager approval notes breaks the chain, enabling instant fraud detection in $\mathcal{O}(N)$ verification time.

### Q4: How does the system handle sudden macroeconomic shocks, such as supply shortages?
> **Answer:** The capacity guardrail enforces $Q_j(P_j) \le 0.98 \times K_j$. When capacity $K_j$ drops significantly (e.g., during supply chain bottlenecks), the SLSQP solver automatically determines the optimal market-clearing price within the $\pm 15\%$ volatility envelope that dampens excess demand without exceeding stock limits or violating margin floors.

### Q5: How is continuous model degradation detected in production?
> **Answer:** The monitoring engine computes the **Population Stability Index (PSI)** and **Wasserstein Distance** between the baseline training distribution and streaming inference windows. A PSI score $\ge 0.25$ indicates significant structural shift and automatically updates Prometheus gauges and triggers retraining alerts.
