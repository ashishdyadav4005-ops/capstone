# Data Dictionary & Provenance Specification

**Project**: BDS-39: Counterfactual Dynamic Pricing with Revenue, Fairness and Customer-Impact Guardrails  
**Dataset Version**: 1.0.0 (Synthetic Confounded Benchmark Dataset)  
**Classification**: Strictly Synthetic / Simulated (No PII, No proprietary secrets)

---

## 1. Schema & Field Definitions

| Column Name | Data Type | Description | Valid Range / Categories | Example |
|---|---|---|---|---|
| `date` | `string (YYYY-MM-DD)` | Calendar date of transaction record | `2023-01-01` to `2023-12-31` | `2023-06-15` |
| `product_id` | `string` | Unique product identifier | `P001`, `P002`, `P003`, `P004`, `P005` | `P001` |
| `location_id` | `string` | Unique regional store/market identifier | `LOC_A`, `LOC_B`, `LOC_C`, `LOC_D` | `LOC_A` |
| `customer_group` | `string` | Customer segment profile | `budget`, `standard`, `premium`, `enterprise` | `standard` |
| `price` | `float` | Unit selling price charged ($) | $> 0.0$ | `47.50` |
| `quantity` | `integer` | Actual realized units sold (capped at capacity) | $\ge 0$ | `74` |
| `capacity` | `integer` | Available stock / inventory limit for period | $> 0$ | `120` |
| `cost` | `float` | Base production / unit procurement cost ($) | $> 0.0$ | `25.00` |
| `holiday` | `integer (0/1)` | Binary indicator for official holiday | `0` or `1` | `1` |
| `event` | `integer (0/1)` | Binary indicator for major local event (concert, game) | `0` or `1` | `0` |
| `weather_temp` | `float` | Local ambient temperature (°C) | $-10.0$ to $45.0$ | `23.4` |
| `competitor_price`| `float` | Observed competitor price for substitute product ($) | $> 0.0$ | `46.00` |
| `unobserved_shock`| `float` | Synthetic unobserved demand shock (latent market buzz) | Normal($0, 1$) | `0.4215` |
| `is_random_policy`| `integer (0/1)` | Flag indicating randomized A/B price exploration day | `0` (concomitant) or `1` (random) | `0` |
| `true_elasticity` | `float` | Ground truth structural price elasticity $\varepsilon^*$ | Negative real (e.g. $-3.0$ to $-0.8$) | `-1.6000` |

---

## 2. Engineered Features

| Feature Name | Data Type | Computation & Leakage Prevention Rule |
|---|---|---|
| `day_of_week` | `integer` | Day of week index (`0` = Monday, `6` = Sunday). |
| `is_weekend` | `integer` | `1` if Saturday or Sunday, `0` otherwise. |
| `month`, `quarter` | `integer` | Calendar month (`1-12`) and quarter (`1-4`). |
| `sin_day_of_year`, `cos_day_of_year` | `float` | Cyclical trigonometric encoding of day of year. |
| `log_price` | `float` | $\log(\text{price})$. |
| `log_quantity` | `float` | $\log(\text{quantity} + 1.0)$. |
| `log_competitor_price` | `float` | $\log(\text{competitor\_price})$. |
| `price_ratio_competitor` | `float` | $\text{price} / \text{competitor\_price}$. |
| `margin_ratio` | `float` | $(\text{price} - \text{cost}) / \text{price}$. |
| `price_lag_1`, `price_lag_7` | `float` | Prior 1-day and 7-day lagged price for the product-location-group. |
| `quantity_lag_1`, `quantity_lag_7` | `float` | Prior 1-day and 7-day realized demand for the product-location-group. |
| `rolling_demand_7d`, `rolling_demand_14d` | `float` | Historical rolling average demand over past 7 and 14 days (**shifted by 1 day to prevent current-day target leakage**). |
| `rolling_price_mean_7d`, `rolling_price_std_7d` | `float` | Historical rolling price mean and volatility over past 7 days (shifted by 1 day). |

---

## 3. Data Generating Process & Confounding Model

The synthetic simulator implements endogeneity where prices in observational periods are adjusted based on demand shifters:

### Pricing Policy (Treatment Assignment):
$$P = P_{\text{base}} \times \left(1 + \gamma_{\text{holiday}} \text{Holiday} + \gamma_{\text{event}} \text{Event} + \gamma_{\text{unobs}} U_{\text{demand}} + \gamma_{\text{comp}} \log\left(\frac{P_{\text{comp}}}{P_{\text{base}}}\right) + \epsilon_{\text{policy}}\right)$$

### Structural Demand Function (Outcome Equation):
$$\log(Q) = \alpha_{\text{product}} + \alpha_{\text{loc}} + \alpha_{\text{group}} + \varepsilon^*_{p, g} \cdot \log\left(\frac{P}{P_{\text{base}}}\right) + \delta_1 \text{Holiday} + \delta_2 \text{Event} + \delta_u U_{\text{demand}} + \eta$$

### Realized Constrained Quantity:
$$Q_{\text{realized}} = \min\left(\lfloor Q \rceil, \text{Capacity}\right)$$

---

## 4. Provenance & Versioning

- **Data Integrity Hash**: Every generated dataset file is hashed with SHA-256 upon generation and saved to metadata tables.
- **Reproducibility**: Parameterized by fixed seeds (`random_seed: 42`).
- **Data Privacy**: 100% synthetic; contains no Personally Identifiable Information (PII) or third-party proprietary data.
