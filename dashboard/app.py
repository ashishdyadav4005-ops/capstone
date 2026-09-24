"""Streamlit Main Application Entrypoint for BDS-39 Dynamic Pricing Decision Support System."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import ApiClient
from dashboard.components.admin_view import render_admin_view
from dashboard.components.analyst_view import render_analyst_view
from dashboard.components.auth_view import render_auth_view
from dashboard.components.governance_view import render_governance_view
from dashboard.components.manager_view import render_manager_view

# 1. Page Configuration
st.set_page_config(
    page_title="Dynamic Pricing Decision Support System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Custom Modern Enterprise Styling
st.markdown(
    """
    <style>
    /* Modern typography and clean layout */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    /* Metric card styling */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    /* Sidebar user profile styling */
    .user-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .role-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background-color: #3B82F6;
        color: #FFFFFF;
    }
    .role-badge.admin { background-color: #EF4444; }
    .role-badge.analyst { background-color: #3B82F6; }
    .role-badge.manager { background-color: #10B981; }
    .role-badge.governance { background-color: #8B5CF6; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_client() -> ApiClient:
    """Initialize or reuse singleton ApiClient instance."""
    token = st.session_state.get("token")
    return ApiClient(token=token)


def main() -> None:
    """Main application loop and role-based view routing."""
    api_client = get_api_client()

    # Check authentication state
    if not st.session_state.get("authenticated", False):
        render_auth_view(api_client)
        return

    # Authenticated user session
    user = st.session_state.get("user_profile", {})
    user_role = user.get("role", "analyst")
    full_name = user.get("full_name", user.get("username", "Staff User"))
    username = user.get("username", "user")

    # Render Sidebar with User Profile & Navigation
    with st.sidebar:
        st.markdown(
            f"""
            <div class="user-card">
                <div style="font-size: 0.85rem; color: #94A3B8;">AUTHENTICATED SESSION</div>
                <div style="font-size: 1.15rem; font-weight: 700; margin: 0.2rem 0;">{full_name}</div>
                <div style="font-size: 0.8rem; color: #CBD5E1; margin-bottom: 0.5rem;">@{username}</div>
                <span class="role-badge {user_role}">{user_role.upper()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🧭 Navigation")

        # Determine accessible views by role
        nav_options: list[str] = []
        if user_role == "admin":
            nav_options = [
                "🔬 Pricing & Simulation (Analyst)",
                "💼 Price Review & Overrides (Manager)",
                "⚖️ Governance & Audit Trail (Governance)",
                "🛡️ User & System Admin",
            ]
        elif user_role == "manager":
            nav_options = [
                "💼 Price Review & Overrides (Manager)",
                "🔬 Pricing & Simulation (Analyst)",
                "⚖️ Governance & Fairness Overview",
            ]
        elif user_role == "governance":
            nav_options = [
                "⚖️ Governance & Cryptographic Audit",
                "🔬 Pricing & Simulation Explorer",
                "💼 Manager Queue Overview",
            ]
        else:  # analyst
            nav_options = [
                "🔬 Pricing & Simulation (Analyst)",
                "📄 Model Card & Governance Summary",
            ]

        selected_view = st.radio("Select Portal View", nav_options, label_visibility="collapsed")

        st.markdown("---")
        backend_status = "🟢 Online" if api_client.is_backend_online() else "⚡ Standalone (Direct DB/Model)"
        st.caption(f"**Engine Mode**: `{backend_status}`")
        st.caption("**System**: `Dynamic Pricing Engine v0.1.0` (TRL 4-5)")

        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["user_profile"] = None
            st.session_state["token"] = None
            st.rerun()

    # Route to selected portal view
    if "Analyst" in selected_view or "Simulation" in selected_view:
        render_analyst_view(api_client)
    elif "Manager" in selected_view or "Overrides" in selected_view:
        render_manager_view(api_client)
    elif "Governance" in selected_view or "Audit" in selected_view or "Fairness" in selected_view:
        render_governance_view(api_client)
    elif "Admin" in selected_view:
        render_admin_view(api_client)
    elif "Model Card" in selected_view:
        render_governance_view(api_client)
    else:
        render_analyst_view(api_client)


if __name__ == "__main__":
    main()
