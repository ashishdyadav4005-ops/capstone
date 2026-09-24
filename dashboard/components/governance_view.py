"""Governance View: Cryptographic Audit Trail, Subgroup Fairness, and Model Transparency."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.api_client import ApiClient


def render_governance_view(api_client: ApiClient) -> None:
    """Render regulatory, compliance, auditability, and fairness oversight interface."""
    st.markdown("## ⚖️ Governance, Fairness & Cryptographic Audit Trail")
    st.caption("Tamper-evident SHA-256 chained audit logs, subgroup fairness disparity metrics, and causal model cards.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔒 Cryptographic Audit Chain",
        "⚖️ Subgroup Fairness Audit",
        "📄 Causal Model Card & Stress Tests",
        "🚨 Emergency Rollback Console",
    ])

    with tab1:
        _render_audit_chain_tab(api_client)

    with tab2:
        _render_fairness_tab(api_client)

    with tab3:
        _render_model_card_tab(api_client)

    with tab4:
        _render_emergency_rollback_tab(api_client)


def _render_audit_chain_tab(api_client: ApiClient) -> None:
    """Tab 1: Cryptographic Audit Trail with live SHA-256 chain verification."""
    st.markdown("### 🔒 Immutable Cryptographic Audit Trail")
    st.caption("Every pricing decision, human override, approval, and security event is sequentially chained with SHA-256 cryptographic hashes ($H_i = \\text{SHA256}(H_{i-1} \\parallel \\text{Entry}_i)$).")

    col_v1, col_v2 = st.columns([2, 1])
    with col_v1:
        st.write("Click below to run a live cryptographic verification across the entire audit database to ensure zero payload tampering, unauthorized deletions, or sequence disruptions.")
    with col_v2:
        verify_btn = st.button("🔍 Verify Cryptographic Chain Integrity", type="primary", use_container_width=True)

    if verify_btn or "last_verification" in st.session_state:
        if verify_btn:
            with st.spinner("Cryptographically verifying SHA-256 hash continuity from genesis block..."):
                v_res = api_client.verify_audit_chain()
                st.session_state["last_verification"] = v_res
        else:
            v_res = st.session_state["last_verification"]

        if v_res.get("is_valid"):
            st.success(
                f"✅ **CRYPTOGRAPHIC CHAIN STATUS: 100% INTACT & VERIFIED**\n\n"
                f"- **Total Entries Verified**: {v_res.get('total_entries')}\n"
                f"- **Chain Tip Hash**: `{v_res.get('latest_entry_hash', 'N/A')}`\n"
                f"- **Verification Timestamp**: `{v_res.get('verified_at')}`"
            )
        else:
            st.error(
                f"❌ **TAMPERING DETECTED AT SEQUENCE #{v_res.get('corrupted_sequence_number')}!**\n\n"
                f"**Diagnostics**: {v_res.get('error_message')}"
            )

    st.markdown("---")
    st.markdown("#### 📜 Recorded Audit Event Stream")
    logs = api_client.get_audit_logs(limit=100)

    if not logs:
        st.info("No audit logs recorded yet.")
        return

    df_logs = pd.DataFrame(logs)
    show_cols = ["sequence_number", "timestamp", "event_type", "actor_username", "actor_role", "entry_hash", "previous_entry_hash"]
    avail_cols = [c for c in show_cols if c in df_logs.columns]
    df_display = df_logs[avail_cols].copy()
    col_map = {
        "sequence_number": "Seq #",
        "timestamp": "Timestamp (UTC)",
        "event_type": "Event Type",
        "actor_username": "Actor",
        "actor_role": "Role",
        "entry_hash": "Entry Hash (SHA-256)",
        "previous_entry_hash": "Previous Hash",
    }
    df_display.rename(columns=col_map, inplace=True)
    st.dataframe(df_display, use_container_width=True, hide_index=True)


def _render_fairness_tab(api_client: ApiClient) -> None:
    """Tab 2: Subgroup Fairness Disparity and Consumer Impact."""
    st.markdown("### ⚖️ Subgroup Price Fairness & Consumer Impact")
    st.caption("Monitors price disparity across customer segments (budget, regular, premium, business) to ensure fairness compliance.")

    fairness_res = api_client.get_fairness_report()
    prods = fairness_res.get("products", [])

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "Overall Fairness Status",
            "COMPLIANT" if fairness_res.get("overall_compliant") else "VIOLATION DETECTED",
            delta="Within 12% Disparity Ceiling" if fairness_res.get("overall_compliant") else "Exceeds Threshold",
        )
    with m2:
        st.metric("Max Observed Price Disparity", f"{fairness_res.get('max_observed_disparity_pct'):.1f}%")
    with m3:
        st.metric("Regulatory Disparity Ceiling", f"{fairness_res.get('threshold_pct'):.1f}%")

    # Plotly Disparity Comparison Chart
    prod_ids = [p["product_id"] for p in prods]
    disparities = [p["max_price_disparity_pct"] for p in prods]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=prod_ids,
        y=disparities,
        marker_color=["#10B981" if d <= 12.0 else "#DC2626" for d in disparities],
        name="Observed Max Group Disparity (%)",
    ))
    fig.add_hline(y=12.0, line_dash="dash", line_color="#DC2626", annotation_text="12% Maximum Allowable Disparity")
    fig.update_layout(
        title="Cross-Segment Price Disparity per Catalog Product",
        yaxis_title="Price Disparity vs Group Average (%)",
        xaxis_title="Product ID",
        height=380,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Group Breakdown
    st.markdown("#### 👥 Subgroup Price Breakdown & Consumer Surplus")
    if prods:
        sel_prod_id = st.selectbox("Inspect Product Details", prod_ids)
        sel_prod_data = next((p for p in prods if p["product_id"] == sel_prod_id), prods[0])
        group_df = pd.DataFrame(sel_prod_data["group_metrics"])
        show_fair_cols = ["customer_group", "optimal_price", "expected_demand", "consumer_surplus_delta_proxy"]
        avail_fair_cols = [c for c in show_fair_cols if c in group_df.columns]
        display_group_df = group_df[avail_fair_cols].copy()
        col_rename_fair = {
            "customer_group": "Customer Segment",
            "optimal_price": "Simulated Price ($)",
            "expected_demand": "Expected Demand",
            "consumer_surplus_delta_proxy": "Consumer Surplus Delta ($)",
        }
        display_group_df.rename(columns=col_rename_fair, inplace=True)
        st.dataframe(display_group_df, use_container_width=True, hide_index=True)


def _render_model_card_tab(api_client: ApiClient) -> None:
    """Tab 3: Model Card, Architecture, Bias Comparison, and Sensitivity Stress Tests."""
    st.markdown("### 📄 Causal Machine Learning Model Card")
    st.caption("Technical transparency metadata, nuisance estimators, bias benchmarks vs OLS, and confounder sensitivity stress testing.")

    card = api_client.get_model_card()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Model Name**: `{card['model_name']}`")
        st.markdown(f"**Architecture**: `{card['model_type']}`")
        st.markdown(f"**Framework**: `{card['framework']}`")
        st.markdown(f"**Cross-Fitting Folds ($K$)**: `{card['cross_fitting_folds']}`")
    with c2:
        st.markdown(f"**Treatment Variable ($T$)**: `{card['target_treatment']}`")
        st.markdown(f"**Outcome Variable ($Y$)**: `{card['target_outcome']}`")
        st.markdown(f"**Governance Status**: `{card['governance_status']}`")

    st.markdown("---")
    st.markdown("#### 🎯 Ground-Truth Elasticity Bias Benchmark")
    st.caption("Demonstrating how Double Machine Learning orthogonalization corrects observational confounding bias compared to naive OLS:")

    bias_df = pd.DataFrame(card["ground_truth_bias_comparison"])
    col_map_bias = {
        "model": "Pricing Model Approach",
        "mean_elasticity": "Mean Estimated Elasticity",
        "bias_vs_truth": "Bias vs Ground Truth (-1.50)",
    }
    bias_df.rename(columns=col_map_bias, inplace=True)
    st.dataframe(bias_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🛡️ Confounder Sensitivity Stress Testing (Rosenbaum $\\gamma$)")
    st.caption("Simulating omitted unobserved confounding strength $\\gamma \\in [0.0, 1.0]$ to verify robustness of causal elasticity estimates:")

    sens_df = pd.DataFrame(card["sensitivity_stress_tests"])
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=sens_df["confounder_strength_gamma"],
        y=sens_df["dml_elasticity"],
        mode="lines+markers",
        name="DML Estimated Elasticity",
        line={"color": "#2563EB", "width": 3},
        marker={"size": 10},
    ))
    fig_sens.add_hline(y=-1.50, line_dash="dash", line_color="#10B981", annotation_text="Ground Truth (-1.50)")
    fig_sens.update_layout(
        title="Elasticity Stability Under Unobserved Confounder Stress",
        xaxis_title="Omitted Confounder Strength (\\gamma)",
        yaxis_title="Estimated Elasticity (\\hat{\\theta})",
        height=380,
        template="plotly_white",
    )
    st.plotly_chart(fig_sens, use_container_width=True)


def _render_emergency_rollback_tab(api_client: ApiClient) -> None:
    """Tab 4: Emergency Rollback Console."""
    st.markdown("### 🚨 Emergency Price Rollback Console")
    st.caption("Governance oversight trigger to instantly restore published catalog prices to safe baseline $P_0$.")

    recs = api_client.list_recommendations(limit=100)
    published_recs = [r for r in recs if r["state"] in ["PUBLISHED", "APPROVED"]]

    if not published_recs:
        st.info("No active published prices eligible for emergency rollback.")
        return

    choices = {
        r["recommendation_id"]: f"{r['recommendation_id']} ({r['product_id']} - {r['customer_group']}) | Active Price: ${r['final_price']:.2f} -> Baseline: ${r['base_price']:.2f}"
        for r in published_recs
    }
    sel_id = st.selectbox("Select Published Price to Rollback", list(choices.keys()), format_func=lambda x: choices[x])
    rollback_notes = st.text_area("Mandatory Emergency Justification", placeholder="Reason for emergency rollback (e.g., unexpected competitor price spike, extreme margin contraction, customer grievance)...")

    if st.button("🚨 EXECUTE EMERGENCY ROLLBACK", type="primary", use_container_width=True):
        if not rollback_notes:
            st.error("Please provide an emergency justification note.")
        else:
            try:
                updated = api_client.rollback_recommendation(sel_id, actor="governance", notes=rollback_notes)
                st.warning(f"EMERGENCY ROLLBACK COMPLETED for `{sel_id}`! Price restored to baseline ${updated['base_price']:.2f}. Audited cryptographically.")
                st.rerun()
            except Exception as e:
                st.error(f"Rollback failed: {e}")
