"""Test script to exercise every dashboard component function and catch any runtime exceptions."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.api_client import ApiClient
from dashboard.components.admin_view import (
    _render_system_config_tab,
    _render_user_management_tab,
)
from dashboard.components.analyst_view import (
    _render_demand_curves_tab,
    _render_portfolio_optimizer_tab,
    _render_what_if_simulator_tab,
)
from dashboard.components.governance_view import (
    _render_audit_chain_tab,
    _render_emergency_rollback_tab,
    _render_fairness_tab,
    _render_model_card_tab,
)
from dashboard.components.manager_view import render_manager_view


def test_all_views():
    client = ApiClient()
    print("ApiClient initialized successfully.")

    # Test Analyst Tabs
    print("Testing _render_demand_curves_tab...")
    _render_demand_curves_tab(client)
    print("PASS: _render_demand_curves_tab")

    print("Testing _render_what_if_simulator_tab...")
    _render_what_if_simulator_tab(client)
    print("PASS: _render_what_if_simulator_tab")

    print("Testing _render_portfolio_optimizer_tab...")
    _render_portfolio_optimizer_tab(client)
    print("PASS: _render_portfolio_optimizer_tab")

    # Test Manager View
    print("Testing render_manager_view...")
    render_manager_view(client)
    print("PASS: render_manager_view")

    # Test Governance Tabs
    print("Testing _render_audit_chain_tab...")
    _render_audit_chain_tab(client)
    print("PASS: _render_audit_chain_tab")

    print("Testing _render_fairness_tab...")
    _render_fairness_tab(client)
    print("PASS: _render_fairness_tab")

    print("Testing _render_model_card_tab...")
    _render_model_card_tab(client)
    print("PASS: _render_model_card_tab")

    print("Testing _render_emergency_rollback_tab...")
    _render_emergency_rollback_tab(client)
    print("PASS: _render_emergency_rollback_tab")

    # Test Admin Tabs
    print("Testing _render_user_management_tab...")
    _render_user_management_tab(client)
    print("PASS: _render_user_management_tab")

    print("Testing _render_system_config_tab...")
    _render_system_config_tab(client)
    print("PASS: _render_system_config_tab")

    print("\nALL DASHBOARD VIEWS AND TABS EXECUTED WITH ZERO ERRORS!")

if __name__ == "__main__":
    test_all_views()
