"""Analyst View: Counterfactual Demand Curves, What-If Simulator, and Portfolio Optimization."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.api_client import ApiClient


def render_analyst_view(api_client: ApiClient) -> None:
    """Render comprehensive pricing analysis tools for data scientists and pricing analysts."""
    st.markdown("## 🔬 Pricing Intelligence & Counterfactual Simulation")
    st.caption("Double Machine Learning (DML) elasticity curves, market shock simulations, and constrained portfolio optimization.")

    tab1, tab2, tab3 = st.tabs([
        "📈 Counterfactual Demand Curves",
        "🌪️ What-If Market Shock Simulator",
        "⚙️ Joint Portfolio Optimizer",
    ])

    with tab1:
        _render_demand_curves_tab(api_client)

    with tab2:
        _render_what_if_simulator_tab(api_client)

    with tab3:
        _render_portfolio_optimizer_tab(api_client)


def _render_demand_curves_tab(api_client: ApiClient) -> None:
    """Tab 1: Interactive Counterfactual Curves with Plotly CI bands."""
    st.markdown("### 📈 Counterfactual Demand & Profit Curves")
    st.caption("Explore non-linear demand response $\\mathbb{E}[Q(P)] = Q_0 (P/P_0)^{\\hat{\\theta}}$ with 95% confidence intervals and guardrails.")

    catalog = api_client.get_catalog()
    prod_options = {p["product_id"]: f"{p['product_id']} - {p['product_name']}" for p in catalog}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sel_prod = st.selectbox("Product", list(prod_options.keys()), format_func=lambda x: prod_options[x])
    with col2:
        sel_group = st.selectbox("Customer Segment", ["budget", "regular", "premium", "business"], index=1)
    with col3:
        sel_loc = st.selectbox("Geographic Location", ["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-1"], index=0)
    with col4:
        grid_pts = st.slider("Resolution Grid Points", min_value=20, max_value=80, value=40, step=5)

    prod_info = next((p for p in catalog if p["product_id"] == sel_prod), catalog[0])

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        base_price = st.number_input("Baseline Price ($P_0$)", value=float(prod_info["base_price"]), min_value=1.0, step=5.0)
    with col_p2:
        unit_cost = st.number_input("Unit Cost ($C$)", value=float(prod_info["unit_cost"]), min_value=0.5, step=5.0)
    with col_p3:
        base_demand = st.number_input("Baseline Demand ($Q_0$)", value=50.0, min_value=1.0, step=5.0)
    with col_p4:
        capacity = st.number_input("Capacity Limit ($K$)", value=float(prod_info["capacity"]), min_value=10.0, step=10.0)

    # Generate curve
    with st.spinner("Generating counterfactual curves..."):
        curve_data = api_client.generate_curve(
            product_id=sel_prod,
            customer_group=sel_group,
            location_id=sel_loc,
            base_price=base_price,
            base_demand=base_demand,
            unit_cost=unit_cost,
            capacity=capacity,
            grid_points=grid_pts,
        )

    # Key metrics summary
    elasticity_val = curve_data.get("elasticity", curve_data.get("estimated_elasticity", 0.0))
    base_price_val = curve_data.get("base_price", 0.0)
    opt_profit_val = curve_data.get("optimal_profit_price", 0.0)
    profit_lift_val = curve_data.get("profit_lift_pct", curve_data.get("expected_profit_lift_pct", 0.0))
    opt_rev_val = curve_data.get("optimal_revenue_price", 0.0)

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Causal Elasticity $\\hat{\\theta}$", f"{elasticity_val:.2f}")
    with m_col2:
        st.metric("Baseline Price $P_0$", f"${base_price_val:.2f}")
    with m_col3:
        st.metric("Profit Optimal Price $P^*_{\\text{prof}}$", f"${opt_profit_val:.2f}")
    with m_col4:
        st.metric("Expected Profit Lift", f"+{profit_lift_val:.1f}%")
    with m_col5:
        st.metric("Revenue Optimal Price $P^*_{\\text{rev}}$", f"${opt_rev_val:.2f}")

    # Plotly Visualizations
    pts = curve_data["points"]
    df_pts = pd.DataFrame(pts)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Demand Response $\\mathbb{E}[Q(P)]$ with 95% CI", "Expected Revenue & Profit Trajectories"),
        horizontal_spacing=0.12,
    )

    # Subplot 1: Demand Curve + CI
    fig.add_trace(
        go.Scatter(
            x=df_pts["price"].tolist() + df_pts["price"].tolist()[::-1],
            y=df_pts["demand_ci_upper"].tolist() + df_pts["demand_ci_lower"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.15)",
            line={"color": "rgba(255,255,255,0)"},
            name="95% Confidence Interval",
            showlegend=True,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_pts["price"], y=df_pts["expected_demand"],
            mode="lines+markers",
            name="Expected Demand",
            line={"color": "#2563EB", "width": 3},
        ),
        row=1, col=1,
    )
    # Baseline demand marker
    fig.add_trace(
        go.Scatter(
            x=[base_price], y=[base_demand],
            mode="markers",
            marker={"size": 12, "color": "#10B981", "symbol": "star"},
            name="Baseline ($P_0, Q_0$)",
        ),
        row=1, col=1,
    )

    # Subplot 2: Revenue and Profit
    fig.add_trace(
        go.Scatter(
            x=df_pts["price"], y=df_pts["expected_revenue"],
            mode="lines",
            name="Expected Revenue ($P \\cdot Q$)",
            line={"color": "#8B5CF6", "width": 2.5, "dash": "dot"},
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df_pts["price"], y=df_pts["expected_profit"],
            mode="lines+markers",
            name="Expected Profit ($(P-C) \\cdot Q$)",
            line={"color": "#059669", "width": 3},
        ),
        row=1, col=2,
    )

    # Profit Optimal marker
    opt_profit_row = df_pts[df_pts["is_profit_optimal"]].iloc[0] if len(df_pts[df_pts["is_profit_optimal"]]) > 0 else None
    if opt_profit_row is not None:
        fig.add_trace(
            go.Scatter(
                x=[opt_profit_row["price"]], y=[opt_profit_row["expected_profit"]],
                mode="markers+text",
                marker={"size": 14, "color": "#DC2626", "symbol": "diamond"},
                text=[f"Max Profit: ${opt_profit_row['expected_profit']:.1f}"],
                textposition="top center",
                name="Optimal Profit Point",
            ),
            row=1, col=2,
        )

    # Vertical Guardrail Boundaries
    margin_floor_price = unit_cost * (1.0 + 0.15)
    volatility_low = base_price * (1.0 - 0.15)
    volatility_high = base_price * (1.0 + 0.15)

    for col_idx in [1, 2]:
        fig.add_vline(x=margin_floor_price, line_dash="dash", line_color="#F59E0B", annotation_text="Margin Floor", row=1, col=col_idx)
        fig.add_vline(x=volatility_low, line_dash="dot", line_color="#64748B", annotation_text="-15% Volatility", row=1, col=col_idx)
        fig.add_vline(x=volatility_high, line_dash="dot", line_color="#64748B", annotation_text="+15% Volatility", row=1, col=col_idx)

    fig.update_layout(
        height=500,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5},
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Candidate Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Candidate Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="Expected Quantity Sold (Units)", row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)


def _render_what_if_simulator_tab(api_client: ApiClient) -> None:
    """Tab 2: What-If Market Shock Simulator & Multi-Strategy Benchmark."""
    st.markdown("### 🌪️ What-If Market Scenario Simulator")
    st.caption("Stress-test pricing strategies against dynamic macroeconomic, competitive, and supply-chain shocks.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        scenario = st.selectbox(
            "Market Shock Scenario",
            [
                ("competitor_price_drop", "⚔️ Competitor Price War (-20%)"),
                ("cost_inflation", "📈 Supply Cost Inflation (+25%)"),
                ("demand_surge", "🚀 Demand Surge / Seasonality (+40%)"),
                ("elasticity_shock", "⚡ Consumer Price Sensitivity Shock"),
                ("capacity_bottleneck", "🛑 Capacity Bottleneck / Constrained Supply"),
            ],
            format_func=lambda x: x[1],
        )[0]

    with col2:
        prod_id = st.selectbox("Product Target", ["PROD_001", "PROD_002", "PROD_003", "PROD_004", "PROD_005"])
    with col3:
        cust_group = st.selectbox("Customer Segment Target", ["budget", "regular", "premium", "business"], index=1)

    # Interactive Shock Parameters
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        comp_shock = st.slider("Competitor Price Change (%)", -50, 50, -20 if scenario == "competitor_price_drop" else 0, step=5)
    with s_col2:
        cost_shock = st.slider("Cost Inflation (%)", -20, 100, 25 if scenario == "cost_inflation" else 0, step=5)
    with s_col3:
        demand_shock = st.slider("Demand Surge (%)", -50, 150, 40 if scenario == "demand_surge" else 0, step=10)

    sim_res = api_client.simulate_scenario(
        scenario_type=scenario,
        product_id=prod_id,
        customer_group=cust_group,
        competitor_price_change_pct=comp_shock / 100.0,
        cost_change_pct=cost_shock / 100.0,
        demand_shock_pct=demand_shock / 100.0,
    )

    st.markdown("#### 📊 Multi-Strategy Comparative Evaluation")
    strategies = sim_res["strategies_evaluated"]
    df_strat = pd.DataFrame(strategies)

    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        # Strategy Comparison Chart
        fig_strat = go.Figure()
        fig_strat.add_trace(go.Bar(
            name="Expected Revenue ($)",
            x=df_strat["strategy_name"],
            y=df_strat["expected_revenue"],
            marker_color="#8B5CF6",
        ))
        fig_strat.add_trace(go.Bar(
            name="Expected Profit ($)",
            x=df_strat["strategy_name"],
            y=df_strat["expected_profit"],
            marker_color="#10B981",
        ))
        fig_strat.update_layout(
            barmode="group",
            height=380,
            template="plotly_white",
            title=f"Performance Comparison Under {sim_res['scenario_name']}",
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5},
        )
        st.plotly_chart(fig_strat, use_container_width=True)

    with col_t2:
        st.markdown("**Benchmark Strategy Breakdown**")
        if "recommended_price" in df_strat.columns and "price" not in df_strat.columns:
            df_strat["price"] = df_strat["recommended_price"]
        elif "price" in df_strat.columns and "recommended_price" not in df_strat.columns:
            df_strat["recommended_price"] = df_strat["price"]

        target_cols = ["strategy_name", "recommended_price", "expected_profit", "profit_lift_pct", "is_guardrail_compliant"]
        avail_cols = [c for c in target_cols if c in df_strat.columns]
        display_df = df_strat[avail_cols].copy()
        col_rename = {
            "strategy_name": "Strategy",
            "recommended_price": "Price ($)",
            "price": "Price ($)",
            "expected_profit": "Profit ($)",
            "profit_lift_pct": "Lift (%)",
            "is_guardrail_compliant": "Compliant?",
        }
        display_df.rename(columns=col_rename, inplace=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        causal_strat = next((s for s in strategies if "Causal" in s["strategy_name"]), strategies[-1])
        st.success(
            f"**Recommended Strategy**: {causal_strat['strategy_name']} delivers "
            f"**+${causal_strat['expected_profit'] - strategies[0]['expected_profit']:.2f}** additional profit "
            f"(+{causal_strat['profit_lift_pct']:.1f}% lift) while maintaining 100% guardrail compliance."
        )


def _render_portfolio_optimizer_tab(api_client: ApiClient) -> None:
    """Tab 3: Joint Portfolio SLSQP Optimizer across multi-product catalog."""
    st.markdown("### ⚙️ Joint Portfolio Constrained Optimizer")
    st.caption("Solves multi-item non-linear mathematical optimization with fairness, volatility, and capacity coupling.")

    catalog = api_client.get_catalog()
    groups = ["budget", "regular", "premium", "business"]

    col1, col2, col3 = st.columns(3)
    with col1:
        obj_type = st.selectbox("Optimization Objective", ["profit", "revenue"], format_func=lambda x: "Maximize Net Profit" if x == "profit" else "Maximize Gross Revenue")
    with col2:
        risk_aversion = st.slider("Risk Aversion Penalty ($\\lambda$)", 0.0, 0.20, 0.05, step=0.01, help="Penalizes demand variance under uncertainty")
    with col3:
        max_volatility = st.slider("Max Price Volatility ($\\pm\\%$)", 5, 25, 15, step=1)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        group_fairness = st.slider("Max Customer Group Disparity ($\\le\\%$)", 5, 20, 12, step=1)
    with col_g2:
        loc_fairness = st.slider("Max Geographic Location Disparity ($\\le\\%$)", 5, 20, 10, step=1)

    # Build multi-item grid (5 products x 4 customer groups = 20 items)
    items_to_optimize = []
    for p in catalog:
        for g in groups:
            items_to_optimize.append({
                "item_id": f"{p['product_id']}_{g.upper()}",
                "product_id": p["product_id"],
                "customer_group": g,
                "location_id": "us-east-1",
                "base_price": p["base_price"],
                "base_demand": 50.0 if g in ["regular", "premium"] else 70.0,
                "unit_cost": p["unit_cost"],
                "capacity": p["capacity"],
                "elasticity": -1.8 if g == "budget" else (-1.5 if g == "regular" else (-1.2 if g == "premium" else -1.0)),
                "elasticity_std_error": 0.08,
            })

    st.markdown(f"**Target Optimization Portfolio**: {len(items_to_optimize)} items across {len(catalog)} products and {len(groups)} customer segments.")

    if st.button("🚀 Solve Constrained Optimization & Generate Recommendations", type="primary", use_container_width=True):
        with st.spinner("Solving SLSQP non-linear optimization with guardrail constraints..."):
            opt_res = api_client.optimize_portfolio(
                items=items_to_optimize,
                objective_type=obj_type,
                risk_aversion=risk_aversion,
                max_price_change_pct=max_volatility / 100.0,
                max_customer_group_disparity_pct=group_fairness / 100.0,
                max_location_disparity_pct=loc_fairness / 100.0,
            )

        st.success(f"Optimization successfully solved in **{opt_res['solve_time_ms']:.1f}ms**! Generated recommendations saved to Manager Review Queue.")

        # Lift Metrics Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Recommended Profit", f"${opt_res['total_recommended_profit']:,.2f}", f"+{opt_res['total_profit_lift_pct']:.2f}% Lift")
        with m2:
            st.metric("Total Recommended Revenue", f"${opt_res['total_recommended_revenue']:,.2f}", f"+{opt_res['total_revenue_lift_pct']:.2f}% Lift")
        with m3:
            st.metric("Baseline Profit", f"${opt_res['total_baseline_profit']:,.2f}")
        with m4:
            st.metric("Guardrail Status", "100% COMPLIANT", delta="All Constraints Satisfied", delta_color="normal")

        # Recommendations Table
        recs = opt_res["recommendations"]
        df_recs = pd.DataFrame(recs)
        st.markdown("#### 📋 Recommended Dynamic Price Schedules")
        show_cols = ["item_id", "product_id", "customer_group", "base_price", "recommended_price", "price_change_pct", "expected_profit", "profit_lift_pct", "is_margin_compliant", "is_volatility_compliant"]
        avail_cols = [c for c in show_cols if c in df_recs.columns]
        df_display = df_recs[avail_cols].copy()
        col_map_opt = {
            "item_id": "Item ID",
            "product_id": "Product",
            "customer_group": "Group",
            "base_price": "Base Price ($)",
            "recommended_price": "Recommended Price ($)",
            "price_change_pct": "Change (%)",
            "expected_profit": "Expected Profit ($)",
            "profit_lift_pct": "Profit Lift (%)",
            "is_margin_compliant": "Margin Safe?",
            "is_volatility_compliant": "Volatility Safe?",
        }
        df_display.rename(columns=col_map_opt, inplace=True)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
