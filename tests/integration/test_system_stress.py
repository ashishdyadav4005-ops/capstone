"""End-to-end system stress testing and resilience verification under extreme conditions."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.lifecycle import RecommendationLifecycleManager
from src.api.main import app
from src.api.repository import RecommendationRepository
from src.api.schemas import RecommendationRecord, RecommendationState
from src.audit.logger import AuditLogger
from src.audit.models import AuditEventType
from src.audit.repository import AuditRepository
from src.audit.verification import AuditChainVerifier
from src.auth.jwt import create_access_token
from src.auth.repository import UserRecord, UserRepository
from src.auth.roles import UserRole
from src.auth.security import hash_password
from src.monitoring.metrics import get_metrics_registry
from src.pricing.guardrails import GuardrailsEngine
from src.pricing.optimizer import (
    ConstrainedPriceOptimizer,
    OptimizationConfig,
    SegmentPricingItem,
)
from src.pricing.simulator import PricingScenario, PricingScenarioSimulator


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def seed_test_users() -> None:
    """Ensure standard test users are available in SQLite for API tests."""
    repo = UserRepository()
    users = [
        ("admin", "Admin@12345", UserRole.ADMIN),
        ("analyst", "Analyst@12345", UserRole.ANALYST),
        ("manager", "Manager@12345", UserRole.MANAGER),
        ("governance", "Governance@12345", UserRole.GOVERNANCE),
    ]
    for uname, pwd, role in users:
        existing = repo.get_by_username(uname)
        if existing is None:
            u = UserRecord(
                user_id=f"USR_{uname.upper()}",
                username=uname,
                email=f"{uname}@company.internal",
                full_name=f"Test {uname.capitalize()}",
                hashed_password=hash_password(pwd),
                role=role,
                is_active=True,
            )
            repo.create_user(u)


@pytest.fixture
def auth_headers():
    """Generate auth headers for each persona role."""
    def _make_header(role: UserRole):
        token = create_access_token({
            "sub": role.value,
            "user_id": f"USR_{role.value.upper()}",
            "role": role.value,
        })
        return {"Authorization": f"Bearer {token}"}

    return _make_header


class TestSystemStressAndResilience:
    """Stress testing suite validating mathematical resilience, concurrency, and security."""

    def test_extreme_macroeconomic_shocks_and_guardrails(self):
        """Verify optimizer and simulator enforce guardrails under extreme macroeconomic shocks."""
        simulator = PricingScenarioSimulator()

        # 1. Hyperinflation scenario (+100% cost surge)
        hyperinflation = PricingScenario(
            name="hyperinflation",
            description="100% cost surge",
            unit_cost_change_pct=1.00,
        )
        res_infl = simulator.compare_strategies(
            product_id="PROD_001",
            base_price=100.0,
            unit_cost=50.0,
            scenario=hyperinflation,
        )
        for ev in res_infl.evaluations:
            if ev.strategy_name == "Causal Guardrailed (Feasible)":
                # Feasible price must adjust to satisfy margin over $100.00 new unit cost
                assert ev.recommended_price >= 115.0 - 1e-2
                assert ev.margin_pct >= 13.0

        # 2. Predatory price war (-70% competitor drop)
        price_war = PricingScenario(
            name="predatory_price_war",
            description="70% competitor price drop",
            competitor_price_change_pct=-0.70,
        )
        res_war = simulator.compare_strategies(
            product_id="PROD_001",
            base_price=100.0,
            unit_cost=60.0,
            scenario=price_war,
        )
        for ev in res_war.evaluations:
            if ev.strategy_name == "Causal Guardrailed (Feasible)":
                # Must not breach -15% volatility bound (floor at $85.00)
                assert ev.recommended_price >= 85.0 - 1e-2
                assert ev.recommended_price >= 60.0 * 1.15

        # 3. Severe capacity destruction (90% bottleneck)
        bottleneck = PricingScenario(
            name="supply_chain_collapse",
            description="90% capacity reduction",
            capacity_multiplier=0.10,
        )
        res_bottle = simulator.compare_strategies(
            product_id="PROD_001",
            base_price=100.0,
            base_demand=50.0,
            unit_cost=60.0,
            capacity=100.0,
            scenario=bottleneck,
        )
        assert len(res_bottle.evaluations) >= 4

    def test_large_scale_portfolio_optimization(self):
        """Stress-test joint SLSQP optimizer on a 60-item portfolio with coupled constraints."""
        optimizer = ConstrainedPriceOptimizer()

        products = [f"PROD_{i:03d}" for i in range(1, 16)] # 15 products
        segments = ["budget", "regular", "premium", "business"] # 4 segments = 60 items

        items: list[SegmentPricingItem] = []
        for p_idx, prod in enumerate(products):
            base_p = 50.0 + (p_idx * 10.0)
            cost = base_p * 0.60
            for seg in segments:
                items.append(
                    SegmentPricingItem(
                        item_id=f"ITEM_{prod}_{seg}",
                        product_id=prod,
                        customer_group=seg,
                        location_id="LOC_URBAN",
                        base_price=base_p,
                        base_demand=40.0,
                        unit_cost=cost,
                        elasticity=-1.2 if seg == "premium" else (-2.0 if seg == "budget" else -1.5),
                        capacity=200.0,
                    )
                )

        cfg = OptimizationConfig(
            objective_type="expected_profit",
            risk_aversion=0.05,
            max_price_change_pct=0.15,
            min_margin_pct_over_cost=0.15,
            max_customer_group_disparity_pct=0.12,
        )

        opt_result = optimizer.optimize(items, config_override=cfg)

        assert opt_result.status in ["OPTIMAL", "FEASIBLE_FALLBACK"]
        assert len(opt_result.recommendations) == 60
        assert opt_result.total_recommended_profit >= opt_result.total_baseline_profit
        assert opt_result.solve_time_ms < 5000.0 # Under 5 seconds for 60 coupled non-linear variables

        # Verify all 60 recommendations strictly respect margin and volatility bounds
        for rec in opt_result.recommendations:
            assert rec.recommended_price >= rec.unit_cost * 1.15 - 1e-4
            assert rec.recommended_price >= rec.base_price * 0.85 - 1e-4
            assert rec.recommended_price <= rec.base_price * 1.15 + 1e-4

    def test_rapid_lifecycle_transitions_and_audit_consistency(self, tmp_path):
        """Test rapid recommendation lifecycle state transitions with verified audit trail."""
        rec_repo = RecommendationRepository(db_path=tmp_path / "stress_recs.db")
        audit_repo = AuditRepository(db_path=tmp_path / "stress_audit.db")
        guardrails = GuardrailsEngine()
        audit_logger = AuditLogger(repository=audit_repo)

        manager = RecommendationLifecycleManager(
            repository=rec_repo,
            guardrails_engine=guardrails,
            audit_logger=audit_logger,
        )

        now = RecommendationRepository.now_iso()
        rec_id = f"STRESS_REC_{now[:10].replace('-', '')}_001"

        # 1. Create initial recommendation
        rec = RecommendationRecord(
            recommendation_id=rec_id,
            item_id="ITEM_STRESS_01",
            product_id="PROD_001",
            customer_group="regular",
            location_id="LOC_URBAN",
            base_price=100.0,
            recommended_price=110.0,
            final_price=110.0,
            unit_cost=60.0,
            expected_demand=45.0,
            expected_revenue=4950.0,
            expected_profit=2250.0,
            margin_pct=45.45,
            state=RecommendationState.GENERATED,
            created_by="OPTIMIZER",
            created_at=now,
            updated_at=now,
        )
        rec_repo.save(rec)

        # 2. Manager Price Override
        overridden = manager.override_price(
            recommendation_id=rec_id,
            override_price=108.0,
            reason="Stress test market adjustment",
            actor="manager",
        )
        assert overridden.state == RecommendationState.UNDER_REVIEW
        assert overridden.final_price == 108.0

        # 3. Approval
        approved = manager.transition(
            recommendation_id=rec_id,
            target_state=RecommendationState.APPROVED,
            actor="manager",
            notes="Approved after override",
        )
        assert approved.state == RecommendationState.APPROVED

        # 4. Publish
        published = manager.transition(
            recommendation_id=rec_id,
            target_state=RecommendationState.PUBLISHED,
            actor="system",
            notes="Published to live catalog",
        )
        assert published.state == RecommendationState.PUBLISHED

        # 5. Emergency Rollback
        rolled_back = manager.rollback(
            recommendation_id=rec_id,
            actor="governance",
            notes="Emergency rollback test",
        )
        assert rolled_back.state == RecommendationState.ROLLED_BACK
        assert rolled_back.final_price == 100.0

        # Verify audit chain remains cryptographically valid after all 5 actions
        verifier = AuditChainVerifier(repository=audit_repo)
        v_result = verifier.verify_chain()
        assert v_result.is_valid is True
        assert v_result.total_entries > 0

    def test_cryptographic_audit_tamper_defense(self, tmp_path):
        """Simulate malicious direct database mutation and verify cryptographic chain detects tampering."""
        audit_repo = AuditRepository(db_path=tmp_path / "tamper_audit.db")
        verifier = AuditChainVerifier(repository=audit_repo)

        # 1. Log legitimate events
        audit_repo.append(
            event_type=AuditEventType.PRICE_RECOMMENDATIONS_GENERATED,
            actor_username="OPTIMIZER",
            actor_role="system",
            action_details={"product_id": "PROD_001", "recommended_price": 112.50},
        )
        audit_repo.append(
            event_type=AuditEventType.PRICE_OVERRIDE_APPLIED,
            actor_username="manager",
            actor_role="manager",
            action_details={"product_id": "PROD_001", "override_price": 110.00},
        )

        # Verify chain is currently clean
        clean_res = verifier.verify_chain()
        assert clean_res.is_valid is True

        # 2. MALICIOUS ATTACK: Directly tamper with the action_details in SQLite
        latest_entry = audit_repo.get_latest_entry()
        assert latest_entry is not None

        with audit_repo._get_connection() as conn:
            # Mutate override price from 110.00 to 125.00 without recomputing hash
            conn.execute(
                """
                UPDATE audit_logs
                SET action_details_json = '{"product_id": "PROD_001", "override_price": 125.00}'
                WHERE entry_id = ?
                """,
                (latest_entry.entry_id,),
            )
            conn.commit()

        # 3. Run verifier to catch tampering
        tamper_res = verifier.verify_chain()
        assert tamper_res.is_valid is False
        assert tamper_res.corrupted_sequence_number == latest_entry.sequence_number
        assert "tampering detected" in (tamper_res.error_message or "").lower() or "mismatch" in (tamper_res.error_message or "").lower()

    def test_end_to_end_drift_monitoring_flow(self, client: TestClient, auth_headers):
        """Verify API drift check flags severe distribution shift and updates Prometheus metrics."""
        headers = auth_headers(UserRole.ANALYST)

        # Generate heavily shifted dataset (extreme inflation)
        np.random.seed(42)
        n = 300
        shifted_records = [
            {
                "price": float(np.random.normal(190.0, 10.0)), # Extreme price shift
                "quantity": int(np.random.poisson(12)), # Demand drop
                "competitor_price": float(np.random.normal(180.0, 10.0)),
                "cost": float(np.random.normal(110.0, 5.0)),
                "customer_group": "premium",
            }
            for _ in range(n)
        ]

        # Call drift check endpoint
        res = client.post(
            "/api/v1/monitoring/drift/check",
            json={"records": shifted_records, "psi_critical_threshold": 0.25},
            headers=headers,
        )
        assert res.status_code == 200
        data = res.json()

        assert data["overall_status"] == "CRITICAL"
        assert data["retraining_recommended"] is True

        # Verify Prometheus gauge was updated with new PSI score
        reg = get_metrics_registry()
        psi_val = reg.feature_drift_psi.get(feature_name="price")
        assert psi_val > 0.25
