"""Synthetic transaction generator with realistic confounding and ground truth price elasticity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ProductProfile:
    product_id: str
    base_cost: float
    base_price: float
    capacity: int
    base_demand: float
    base_elasticity: float


@dataclass
class LocationProfile:
    location_id: str
    demand_multiplier: float
    price_sensitivity: float
    avg_temp: float


# Segment-specific elasticity modifiers
CUSTOMER_GROUP_ELASTICITY_MODIFIERS: dict[str, float] = {
    "budget": -0.8,      # More price-sensitive (e.g. -2.3 if base is -1.5)
    "standard": 0.0,     # Baseline sensitivity
    "premium": 0.5,      # Less price-sensitive (e.g. -1.0 if base is -1.5)
    "enterprise": 0.7,   # Highly inelastic (e.g. -0.8 if base is -1.5)
}

CUSTOMER_GROUP_VOLUME_SHARE: dict[str, float] = {
    "budget": 0.30,
    "standard": 0.40,
    "premium": 0.20,
    "enterprise": 0.10,
}

DEFAULT_PRODUCTS: list[ProductProfile] = [
    ProductProfile("P001", base_cost=25.0, base_price=45.0, capacity=120, base_demand=85.0, base_elasticity=-1.6),
    ProductProfile("P002", base_cost=40.0, base_price=75.0, capacity=90, base_demand=60.0, base_elasticity=-1.8),
    ProductProfile("P003", base_cost=15.0, base_price=30.0, capacity=180, base_demand=130.0, base_elasticity=-2.2),
    ProductProfile("P004", base_cost=65.0, base_price=120.0, capacity=60, base_demand=40.0, base_elasticity=-1.2),
    ProductProfile("P005", base_cost=10.0, base_price=22.0, capacity=250, base_demand=190.0, base_elasticity=-2.5),
]

DEFAULT_LOCATIONS: list[LocationProfile] = [
    LocationProfile("LOC_A", demand_multiplier=1.25, price_sensitivity=0.90, avg_temp=22.0),
    LocationProfile("LOC_B", demand_multiplier=1.00, price_sensitivity=1.00, avg_temp=18.0),
    LocationProfile("LOC_C", demand_multiplier=0.80, price_sensitivity=1.15, avg_temp=15.0),
    LocationProfile("LOC_D", demand_multiplier=1.10, price_sensitivity=0.95, avg_temp=28.0),
]


class SyntheticDataGenerator:
    """Generates synthetic sales records with realistic endogeneity, confounders, and known causal elasticity."""

    def __init__(
        self,
        products: list[ProductProfile] | None = None,
        locations: list[LocationProfile] | None = None,
        start_date: str = "2023-01-01",
        n_days: int = 365,
        random_seed: int = 42,
        random_policy_ratio: float = 0.15,
        unobserved_confounder_strength: float = 0.35,
    ):
        self.products = products or DEFAULT_PRODUCTS
        self.locations = locations or DEFAULT_LOCATIONS
        self.start_date = pd.to_datetime(start_date)
        self.n_days = n_days
        self.random_seed = random_seed
        self.random_policy_ratio = random_policy_ratio
        self.unobserved_confounder_strength = unobserved_confounder_strength

    def generate(self) -> pd.DataFrame:
        """Generate full synthetic transactions dataframe."""
        rng = np.random.default_rng(self.random_seed)
        date_range = pd.date_range(start=self.start_date, periods=self.n_days, freq="D")

        records: list[dict[str, Any]] = []

        # Generate global environmental factors per day
        daily_context = {}
        for day in date_range:
            is_weekend = int(day.dayofweek in [5, 6])
            # Holidays (arbitrary sample dates throughout year)
            is_holiday = int(day.day in [1, 15] and day.month in [1, 7, 11, 12])
            # Major event (sports/concert/festival, roughly ~5% chance)
            is_event = int(rng.random() < 0.05)
            # Unobserved macro demand shock (market hype / word-of-mouth)
            unobserved_shock = rng.normal(0, 1.0)

            daily_context[day] = {
                "holiday": is_holiday,
                "event": is_event,
                "is_weekend": is_weekend,
                "unobserved_shock": unobserved_shock,
            }

        for day in date_range:
            ctx = daily_context[day]

            for loc in self.locations:
                # Weather temperature fluctuation
                weather_temp = loc.avg_temp + 5.0 * np.sin(2 * np.pi * day.dayofyear / 365.0) + rng.normal(0, 2.5)

                for prod in self.products:
                    # Competitor pricing based on cost with noise
                    base_comp_price = prod.base_price * (1.0 + rng.normal(0, 0.08))
                    competitor_price = max(prod.base_cost * 1.05, round(base_comp_price, 2))

                    for grp, grp_share in CUSTOMER_GROUP_VOLUME_SHARE.items():
                        # Calculate True Ground Truth Elasticity for this product-segment combination
                        grp_mod = CUSTOMER_GROUP_ELASTICITY_MODIFIERS[grp]
                        true_elasticity = prod.base_elasticity * loc.price_sensitivity + grp_mod

                        # Demand shifters (Observed + Unobserved)
                        holiday_effect = 0.35 * ctx["holiday"]
                        event_effect = 0.50 * ctx["event"]
                        weekend_effect = 0.20 * ctx["is_weekend"]
                        weather_effect = 0.015 * (weather_temp - loc.avg_temp)
                        competitor_effect = 0.25 * np.log(competitor_price / prod.base_price)
                        unobs_effect = self.unobserved_confounder_strength * ctx["unobserved_shock"]

                        # Overall latent demand index
                        latent_demand_shifter = (
                            holiday_effect + event_effect + weekend_effect +
                            weather_effect + competitor_effect + unobs_effect
                        )

                        # CONFOUNDED PRICING POLICY:
                        # In observational policy, company raises prices when demand shifters are high (holiday, events, unobserved shock)
                        is_random_policy_day = rng.random() < self.random_policy_ratio

                        if is_random_policy_day:
                            # Randomized price experiment (unconfounded exploration)
                            price_multiplier = 1.0 + rng.uniform(-0.18, 0.18)
                        else:
                            # Standard confounded policy: price is set in response to observed & unobserved demand shifters
                            policy_adjustment = (
                                0.30 * ctx["holiday"] +
                                0.35 * ctx["event"] +
                                0.15 * ctx["is_weekend"] +
                                0.25 * unobs_effect + # Endogeneity: pricing responds to unobserved shock!
                                0.20 * np.log(competitor_price / prod.base_price) +
                                rng.normal(0, 0.03) # Policy noise
                            )
                            price_multiplier = 1.0 + policy_adjustment

                        # Ensure price respects margin over cost
                        raw_price = prod.base_price * price_multiplier
                        price = max(prod.base_cost * 1.10, round(raw_price, 2))

                        # STRUCTURAL DEMAND FUNCTION:
                        # log(Q) = log(BaseDemand) + log(LocMultiplier) + log(GrpShare) + Elasticity * log(Price / BasePrice) + DemandShifters + Noise
                        log_base = (
                            np.log(prod.base_demand) +
                            np.log(loc.demand_multiplier) +
                            np.log(grp_share)
                        )

                        log_price_ratio = np.log(price / prod.base_price)
                        demand_noise = rng.normal(0, 0.10) # idiosyncratic transaction noise

                        log_expected_demand = (
                            log_base +
                            true_elasticity * log_price_ratio +
                            latent_demand_shifter +
                            demand_noise
                        )

                        expected_demand = np.exp(log_expected_demand)

                        # Subgroup allocated capacity
                        subgroup_capacity = max(5, int(round(prod.capacity * grp_share * loc.demand_multiplier)))

                        # Actual realized quantity sold is capped by available capacity
                        realized_quantity = int(min(round(expected_demand), subgroup_capacity))
                        realized_quantity = max(0, realized_quantity)

                        records.append({
                            "date": day.strftime("%Y-%m-%d"),
                            "product_id": prod.product_id,
                            "location_id": loc.location_id,
                            "customer_group": grp,
                            "price": float(price),
                            "quantity": int(realized_quantity),
                            "capacity": int(subgroup_capacity),
                            "cost": float(prod.base_cost),
                            "holiday": int(ctx["holiday"]),
                            "event": int(ctx["event"]),
                            "weather_temp": float(round(weather_temp, 1)),
                            "competitor_price": float(competitor_price),
                            "unobserved_shock": float(round(ctx["unobserved_shock"], 4)),
                            "is_random_policy": int(is_random_policy_day),
                            "true_elasticity": float(round(true_elasticity, 4)),
                        })

        df = pd.DataFrame(records)
        return df
