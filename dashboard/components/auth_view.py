"""Authentication View for BDS-39 Streamlit Decision Support Portal."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import ApiClient


def render_auth_view(api_client: ApiClient) -> None:
    """Render modern login screen with 1-click role presets for evaluation."""
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2.2rem; padding: 1.6rem 1rem; border-radius: 1rem; background: linear-gradient(135deg, rgba(37, 99, 235, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%); border: 1px solid rgba(148, 163, 184, 0.2);">
            <h1 style="font-size: 2.3rem; font-weight: 800; background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.35rem;">
                📈 Counterfactual Dynamic Pricing Engine
            </h1>
            <p style="font-size: 1.05rem; color: #94A3B8; margin: 0; font-weight: 500;">
                Decision Support System with Revenue, Fairness & Capacity Guardrails
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🔐 Staff Authentication")
        st.caption("Enter your authorized staff credentials or select a quick-login persona.")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="e.g. analyst, manager, admin, governance")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submit_btn = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submit_btn:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    with st.spinner("Authenticating credentials..."):
                        res = api_client.login(username, password)
                        if res.get("success"):
                            data = res["data"]
                            st.session_state["authenticated"] = True
                            st.session_state["user_profile"] = data
                            st.session_state["token"] = data.get("access_token")
                            st.success(f"Welcome, {data.get('full_name')} ({data.get('role').upper()})!")
                            st.rerun()
                        else:
                            st.error(res.get("error", "Authentication failed."))

        st.markdown("---")
        st.markdown("#### ⚡ 1-Click Evaluation Personas")
        st.caption("Instant demo access with pre-configured role permissions:")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if st.button("🔬 Analyst Persona", use_container_width=True, help="Explore counterfactual curves & what-if simulator"):
                _quick_login(api_client, "analyst", "Analyst@12345")

            if st.button("💼 Manager Persona", use_container_width=True, help="Review recommendations, apply price overrides & approvals"):
                _quick_login(api_client, "manager", "Manager@12345")

        with p_col2:
            if st.button("⚖️ Governance Officer", use_container_width=True, help="Cryptographic audit chain verification & fairness audits"):
                _quick_login(api_client, "governance", "Governance@12345")

            if st.button("🛡️ System Admin", use_container_width=True, help="User management, account unlock & system configurations"):
                _quick_login(api_client, "admin", "Admin@12345")


def _quick_login(api_client: ApiClient, username: str, password: str) -> None:
    """Helper to perform instant 1-click login."""
    with st.spinner(f"Signing in as {username}..."):
        res = api_client.login(username, password)
        if res.get("success"):
            data = res["data"]
            st.session_state["authenticated"] = True
            st.session_state["user_profile"] = data
            st.session_state["token"] = data.get("access_token")
            st.rerun()
        else:
            st.error(f"Failed to login as {username}: {res.get('error')}")
