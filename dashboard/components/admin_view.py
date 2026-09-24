"""Admin View: User Management, Account Lockout Controls, and System Configurations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.api_client import ApiClient
from src.auth.repository import UserRecord
from src.auth.roles import UserRole
from src.auth.security import hash_password
from src.common.config import get_config


def render_admin_view(api_client: ApiClient) -> None:
    """Render system administrator controls for user accounts and server configurations."""
    st.markdown("## 🛡️ System Administration & User Management")
    st.caption("Manage staff user accounts, lockout policies, and inspect active engine configurations.")

    tab1, tab2 = st.tabs(["👥 User Accounts & Access Control", "⚙️ System Configuration & Health"])

    with tab1:
        _render_user_management_tab(api_client)

    with tab2:
        _render_system_config_tab(api_client)


def _render_user_management_tab(api_client: ApiClient) -> None:
    """Tab 1: User list, account unlock, and creation."""
    st.markdown("### 👥 Authorized Staff Users")
    users = api_client.list_users()
    df_users = pd.DataFrame(users)

    show_cols = ["user_id", "username", "full_name", "email", "role", "is_active", "failed_login_attempts", "locked_until"]
    avail_cols = [c for c in show_cols if c in df_users.columns]
    df_display = df_users[avail_cols].copy()
    col_map_users = {
        "user_id": "User ID",
        "username": "Username",
        "full_name": "Full Name",
        "email": "Email",
        "role": "Role",
        "is_active": "Active?",
        "failed_login_attempts": "Failed Logins",
        "locked_until": "Locked Until",
    }
    df_display.rename(columns=col_map_users, inplace=True)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    # 1. Unlock Account
    with col1:
        st.markdown("#### 🔓 Unlock User Account")
        locked_users = [u for u in users if u.get("locked_until") is not None or u.get("failed_login_attempts", 0) >= 5]
        if not locked_users:
            st.info("No accounts are currently locked.")
        else:
            sel_user_id = st.selectbox("Select Locked Account", [u["user_id"] for u in locked_users], format_func=lambda x: f"{x} ({next(u['username'] for u in locked_users if u['user_id'] == x)})")
            if st.button("Unlock Account Now", type="primary", use_container_width=True):
                success = api_client.unlock_user(sel_user_id)
                if success:
                    st.success(f"User `{sel_user_id}` successfully unlocked!")
                    st.rerun()
                else:
                    st.error("Failed to unlock user.")

    # 2. Create User
    with col2:
        st.markdown("#### ➕ Create New Staff User")
        with st.form("create_user_form"):
            new_uname = st.text_input("Username")
            new_name = st.text_input("Full Name")
            new_email = st.text_input("Email")
            new_pwd = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["analyst", "manager", "governance", "admin"])
            submit_user = st.form_submit_button("Create User Account", use_container_width=True)

            if submit_user:
                if not new_uname or not new_pwd:
                    st.error("Username and password are required.")
                else:
                    try:
                        repo = api_client.user_repo
                        if repo.get_by_username(new_uname):
                            st.error(f"Username '{new_uname}' already exists.")
                        else:
                            u = UserRecord(
                                user_id=f"USR_{new_uname.upper()[:8]}",
                                username=new_uname.strip().lower(),
                                email=new_email,
                                full_name=new_name,
                                hashed_password=hash_password(new_pwd),
                                role=UserRole(new_role),
                                is_active=True,
                            )
                            repo.create_user(u)
                            st.success(f"User account `{new_uname}` ({new_role.upper()}) created!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create user: {e}")


def _render_system_config_tab(api_client: ApiClient) -> None:
    """Tab 2: Active engine configurations and guardrails."""
    st.markdown("### ⚙️ Engine Configurations & Active Guardrails")
    config = get_config()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🖥️ Server & Runtime Environment")
        st.write(f"- **Application Name**: `{config.app.name}`")
        st.write(f"- **Environment**: `{config.app.environment}`")
        st.write(f"- **Backend API URL**: `{api_client.base_url}`")
        st.write(f"- **Backend Service Online?**: `{'ONLINE' if api_client.is_backend_online() else 'STANDALONE (Direct Model/DB Mode)'}`")
        st.write(f"- **Database Path**: `{config.database.url}`")
        st.write(f"- **Log Level**: `{config.logging.level}`")

    with c2:
        st.markdown("#### 🛡️ Active Guardrail Constraints")
        config.raw_configs.get("guardrails", {})
        guardrails_data = [
            {"Guardrail Constraint": "Price Volatility Floor & Ceiling", "Limit": "±15.0% vs Baseline Price P0"},
            {"Guardrail Constraint": "Minimum Margin Percentage", "Limit": "Cost + 15.0%"},
            {"Guardrail Constraint": "Absolute Minimum Margin Currency", "Limit": "Cost + $2.00 / Unit"},
            {"Guardrail Constraint": "Customer Group Fairness Ceiling", "Limit": "≤12.0% Maximum Disparity"},
            {"Guardrail Constraint": "Geographic Location Fairness Ceiling", "Limit": "≤10.0% Maximum Disparity"},
            {"Guardrail Constraint": "Capacity Utilization Threshold", "Limit": "≤98.0% Inventory Limit"},
        ]
        st.dataframe(pd.DataFrame(guardrails_data), use_container_width=True, hide_index=True)
