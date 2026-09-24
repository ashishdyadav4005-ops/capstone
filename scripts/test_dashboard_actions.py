"""Extended test script to exercise all dashboard actions, buttons, and state mutations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.api_client import ApiClient


def test_dashboard_actions():
    client = ApiClient()
    print("Testing client actions...")

    # 1. Login
    login_res = client.login("admin", "Admin@12345")
    assert login_res["success"] is True, f"Admin login failed: {login_res}"
    print("PASS: Admin login")

    # 2. List recommendations
    recs = client.list_recommendations()
    print(f"PASS: list_recommendations ({len(recs)} records found)")

    # 3. Optimize portfolio
    catalog = client.get_catalog()
    items = []
    for p in catalog:
        for g in ["budget", "regular", "premium", "business"]:
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
    opt_res = client.optimize_portfolio(items=items)
    assert "recommendations" in opt_res
    print(f"PASS: optimize_portfolio (generated {len(opt_res['recommendations'])} recs in {opt_res['solve_time_ms']:.1f}ms)")

    # 4. Price override
    rec_id = opt_res["recommendations"][0]["recommendation_id"]
    ov_res = client.override_price(rec_id=rec_id, override_price=95.0, reason="Test business waiver override", actor="manager")
    assert ov_res["override_price"] == 95.0
    print("PASS: override_price")

    # 5. Approve recommendation
    app_res = client.approve_recommendation(rec_id=rec_id, actor="manager", notes="Approved via automated test")
    assert app_res["state"] == "APPROVED"
    print("PASS: approve_recommendation")

    # 6. Publish recommendation
    pub_res = client.publish_recommendation(rec_id=rec_id, actor="system")
    assert pub_res["state"] == "PUBLISHED"
    print("PASS: publish_recommendation")

    # 7. Rollback recommendation
    rb_res = client.rollback_recommendation(rec_id=rec_id, actor="governance", notes="Emergency rollback test")
    assert rb_res["state"] == "ROLLED_BACK"
    print("PASS: rollback_recommendation")

    # 8. Cryptographic audit chain verification
    v_res = client.verify_audit_chain()
    assert v_res["is_valid"] is True, f"Audit verification failed: {v_res}"
    print(f"PASS: verify_audit_chain (Verified {v_res['total_entries']} blocks with tip {v_res['latest_entry_hash'][:16]}...)")

    # 9. Fairness report
    fair_res = client.get_fairness_report()
    assert "overall_compliant" in fair_res
    print(f"PASS: get_fairness_report (Observed max disparity: {fair_res['max_observed_disparity_pct']}%)")

    # 10. Model card
    card = client.get_model_card()
    assert "model_name" in card
    print(f"PASS: get_model_card ({card['model_name']})")

    # 11. User management
    users = client.list_users()
    assert len(users) >= 4
    print(f"PASS: list_users ({len(users)} staff accounts)")

    print("\n>>> ALL API CLIENT ACTIONS & DATA MUTATIONS PASSED PERFECTLY! <<<")

if __name__ == "__main__":
    test_dashboard_actions()
