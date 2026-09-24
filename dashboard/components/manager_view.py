"""Manager View: Recommendation Review Queue, Price Overrides, and Approvals."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.api_client import ApiClient


def render_manager_view(api_client: ApiClient) -> None:
    """Render management portal for price review, human-in-the-loop overrides, and catalog publishing."""
    st.markdown("## 💼 Price Recommendation Review & Decision Console")
    st.caption("Human-in-the-loop review queue for approving, overriding, and publishing dynamic price schedules.")

    recs = api_client.list_recommendations(limit=200)

    if not recs:
        st.info("No recommendations found in the database. Please visit the **Analyst Portal** to generate an optimization batch first.")
        if st.button("Generate Demo Optimization Batch Now", type="primary"):
            _generate_demo_recs(api_client)
            st.rerun()
        return

    df_recs = pd.DataFrame(recs)

    # 1. Summary Statistics Cards
    total_recs = len(df_recs)
    pending_count = len(df_recs[df_recs["state"].isin(["GENERATED", "UNDER_REVIEW"])])
    approved_count = len(df_recs[df_recs["state"] == "APPROVED"])
    published_count = len(df_recs[df_recs["state"] == "PUBLISHED"])
    avg_margin = df_recs["margin_pct"].mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Recommendations", total_recs)
    with m2:
        st.metric("Pending Review", pending_count, delta=f"{pending_count} Actionable" if pending_count > 0 else "All Reviewed")
    with m3:
        st.metric("Approved", approved_count)
    with m4:
        st.metric("Published to Catalog", published_count)
    with m5:
        st.metric("Average Margin", f"{avg_margin:.1f}%")

    st.markdown("---")

    # 2. Review Queue Table with Filters
    st.markdown("### 📋 Recommendation Review Queue")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        filter_prod = st.selectbox("Filter Product", ["All"] + sorted(df_recs["product_id"].unique()))
    with f_col2:
        filter_group = st.selectbox("Filter Customer Segment", ["All"] + sorted(df_recs["customer_group"].unique()))
    with f_col3:
        filter_state = st.selectbox("Filter Lifecycle State", ["All", "GENERATED", "UNDER_REVIEW", "APPROVED", "PUBLISHED", "REJECTED", "ROLLED_BACK"])

    filtered_df = df_recs.copy()
    if filter_prod != "All":
        filtered_df = filtered_df[filtered_df["product_id"] == filter_prod]
    if filter_group != "All":
        filtered_df = filtered_df[filtered_df["customer_group"] == filter_group]
    if filter_state != "All":
        filtered_df = filtered_df[filtered_df["state"] == filter_state]

    display_cols = [
        "recommendation_id", "product_id", "customer_group", "location_id",
        "base_price", "recommended_price", "final_price", "unit_cost",
        "margin_pct", "expected_profit", "state", "override_price", "reviewed_by",
    ]
    avail_mgr_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(filtered_df[avail_mgr_cols], use_container_width=True, hide_index=True)

    st.markdown("---")

    # 3. Action Columns: Overrides & Single/Batch Operations
    act_col1, act_col2 = st.columns([1, 1])

    with act_col1:
        st.markdown("### ✏️ Human-in-the-Loop Price Override")
        st.caption("Manually adjust price with guardrail safety checks and mandatory audit justification.")

        rec_choices = {
            r["recommendation_id"]: f"{r['recommendation_id']} ({r['product_id']} - {r['customer_group']}) | Cur: ${r['final_price']:.2f} | State: {r['state']}"
            for r in recs
        }
        sel_rec_id = st.selectbox("Select Recommendation to Override", list(rec_choices.keys()), format_func=lambda x: rec_choices[x])
        sel_rec = next((r for r in recs if r["recommendation_id"] == sel_rec_id), recs[0])

        with st.form("override_form"):
            st.write(f"**Item**: `{sel_rec['item_id']}` | **Base Price**: `${sel_rec['base_price']:.2f}` | **Unit Cost**: `${sel_rec['unit_cost']:.2f}`")
            new_price = st.number_input("New Override Price ($)", value=float(sel_rec["final_price"]), min_value=1.0, step=1.0)
            override_reason = st.text_area("Mandatory Business Justification", placeholder="Explain why the algorithmic price is being adjusted (e.g., promotional agreement, VIP client waiver)...")
            submit_override = st.form_submit_button("Submit Price Override", type="primary", use_container_width=True)

            if submit_override:
                if not override_reason or len(override_reason.strip()) < 5:
                    st.error("A clear business justification reason (>= 5 characters) is required.")
                else:
                    try:
                        api_client.override_price(
                            rec_id=sel_rec_id,
                            override_price=new_price,
                            reason=override_reason,
                            actor="manager",
                        )
                        st.success(f"Price for `{sel_rec_id}` overridden to ${new_price:.2f} (State: UNDER_REVIEW). Audited successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Price override rejected by guardrail engine: {e}")

    with act_col2:
        st.markdown("### 🚀 Lifecycle State Transitions")
        st.caption("Promote recommendations through approval and live production publishing.")

        st.write(f"**Target Recommendation**: `{sel_rec_id}` (Current State: `{sel_rec['state']}`)")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("✅ Approve Price", use_container_width=True, help="Mark as APPROVED for deployment"):
                try:
                    api_client.approve_recommendation(sel_rec_id, actor="manager", notes="Manager approved via UI")
                    st.success(f"Recommendation `{sel_rec_id}` APPROVED!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Approval failed: {e}")

            if st.button("📢 Publish to Catalog", use_container_width=True, type="primary", help="Publish approved price live to customer catalog"):
                try:
                    api_client.publish_recommendation(sel_rec_id, actor="manager")
                    st.success(f"Recommendation `{sel_rec_id}` PUBLISHED live to customer pricing catalog!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Publish failed: {e}")

        with btn_col2:
            reject_reason = st.text_input("Rejection Reason", placeholder="Reason for rejection...", key="rej_reason")
            if st.button("❌ Reject Price", use_container_width=True, help="Reject recommendation"):
                if not reject_reason:
                    st.error("Please provide a rejection reason.")
                else:
                    try:
                        api_client.reject_recommendation(sel_rec_id, reason=reject_reason, actor="manager")
                        st.warning(f"Recommendation `{sel_rec_id}` REJECTED.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Rejection failed: {e}")

            if st.button("⏪ Emergency Rollback", use_container_width=True, help="Instantly restore baseline price P0"):
                try:
                    api_client.rollback_recommendation(sel_rec_id, actor="manager", notes="Emergency rollback requested by manager")
                    st.warning(f"Recommendation `{sel_rec_id}` ROLLED BACK to baseline price ${sel_rec['base_price']:.2f}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Rollback failed: {e}")

        st.markdown("---")
        st.markdown("#### ⚡ Batch Actions")
        if st.button("Approve All Pending Recommendations", use_container_width=True):
            pending_items = [r for r in recs if r["state"] in ["GENERATED", "UNDER_REVIEW"]]
            approved_count = 0
            for r in pending_items:
                try:
                    api_client.approve_recommendation(r["recommendation_id"], actor="manager", notes="Batch approved by manager")
                    approved_count += 1
                except Exception:
                    pass
            st.success(f"Successfully batch-approved {approved_count} recommendations!")
            st.rerun()


def _generate_demo_recs(api_client: ApiClient) -> None:
    """Helper to populate demo recommendations."""
    catalog = api_client.get_catalog()
    groups = ["budget", "regular", "premium", "business"]
    items = []
    for p in catalog:
        for g in groups:
            items.append({
                "item_id": f"{p['product_id']}_{g.upper()}",
                "product_id": p["product_id"],
                "customer_group": g,
                "location_id": "us-east-1",
                "base_price": p["base_price"],
                "base_demand": 50.0,
                "unit_cost": p["unit_cost"],
                "capacity": p["capacity"],
                "elasticity": -1.5,
                "elasticity_std_error": 0.08,
            })
    api_client.optimize_portfolio(items=items)
