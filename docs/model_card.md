# Model Card: BDS-39 Double Machine Learning Causal Elasticity Engine

**Model Architecture**: Robinson's Double Machine Learning (DML) with $K$-Fold Cross-Fitting  
**Version**: 1.0.0  
**Status**: Production Candidate  
**Intended Deployment**: Internal Decision-Support System for Revenue Managers

---

## 1. Model Overview & Purpose
The BDS-39 Causal Elasticity Engine estimates heterogeneous, unconfounded price elasticities of demand:
$$\theta(W) = \frac{\partial \mathbb{E}[\log(Q) \mid \text{do}(P), W]}{\partial \log(P)}$$

Unlike standard observational regressions that suffer from endogeneity bias (prices being higher during peak holiday/event demand), this model uses orthogonalized cross-fitting nuisance models $\hat{g}(X)$ (demand shifters) and $\hat{m}(X)$ (pricing policy) to isolate the true causal response to price interventions.

---

## 2. Intended Use & Scope Exclusions
- **Primary Use**: Estimating counterfactual demand curves across candidate prices ($[-25\%, +25\%]$) to feed into the SciPy constrained revenue optimization engine.
- **Out of Scope**: 
  - Direct customer-facing automated price setting without human-in-the-loop review.
  - High-frequency tick-by-tick micro-trading.
  - Extrapolation beyond observed operational price intervals.

---

## 3. Training & Validation Data
- **Dataset**: BDS-39 Synthetic Confounded Benchmark Dataset (29,200 transactions across 5 products, 4 locations, 4 customer tiers over 365 days).
- **Splits**: Time-based chronological split (Train: Days 1–255, Val: Days 256–310, Test: Days 311–365). Zero data leakage verified.
- **Features**: Lagged demand, rolling demand trends (7d/14d shifted by 1 day), competitor prices, weather temperature, holiday & event indicators, cyclical seasonality.

---

## 4. Quantitative Evaluation & Benchmarks

| Metric | Naive Non-Causal Baseline | Causal DML (Ours) | Target Quality Gate |
|---|---|---|---|
| **Out-of-Sample Demand MAPE** | $14.2\%$ | **$11.8\%$** | $\le 15.0\%$ |
| **Elasticity Estimation Bias vs Ground Truth** | $0.85$ (underestimated) | **$0.08$** | $\le 0.15$ |
| **Bias Reduction vs Naive** | Baseline | **$> 85\%$ Reduction** | $\ge 50.0\%$ |
| **95% Interval Empirical Coverage** | N/A | **$94.6\%$** | $90\% - 97\%$ |
| **Elasticity Sign Integrity** | Inconsistent | **$100\%$ Negative** | $100\%$ Negative |

---

## 5. Subgroup Performance & Fairness Diagnostics
- **Customer Segments**:
  - `budget`: High elasticity ($\approx -2.4$), highly sensitive to price increases.
  - `standard`: Moderate elasticity ($\approx -1.6$).
  - `premium`: Low elasticity ($\approx -1.1$).
  - `enterprise`: Inelastic ($\approx -0.9$).
- **Fairness Protection**: Model predictions are paired with strict business guardrails (max $10\%$ location price disparity, max $12\%$ customer group disparity) to prevent algorithmic gouging.

---

## 6. Sensitivity Analysis & Robustness
- **Omitted Variable Stress Test**: Omitting the primary calendar demand shifter (`holiday`) shifts estimated elasticity by only $6.2\%$, retaining strong negative directionality.
- **Synthetic Shock Injection**: The causal elasticity sign remains robust and negative under synthetic unobserved confounding up to $\gamma = 0.75$.

---

## 7. Model Governance & Limitations
- **Identification Assumptions**: Relies on conditional unconfoundedness given rich historical context, competitor tracking, and price exploration periods.
- **Cold-Start Policy**: Unseen products or locations fall back to conservative global elasticity point estimates with widened uncertainty bounds.
