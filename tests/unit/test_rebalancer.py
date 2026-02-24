"""Tests for marginal delta rebalancing at window transitions."""

from __future__ import annotations

import math
from datetime import date

import pytest

from stat_arb.config.constants import OrderSide, PositionDirection, RebalanceAction
from stat_arb.config.settings import SizingConfig
from stat_arb.discovery.pair_filter import QualifiedPair
from stat_arb.execution.rebalancer import (
    InventoryRebalancer,
    OpenPositionView,
    RebalanceResult,
)


def _make_pair(
    symbol_y: str = "AAA",
    symbol_x: str = "BBB",
    hedge_ratio: float = 1.5,
    **overrides: object,
) -> QualifiedPair:
    defaults = dict(
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        sector="tech",
        formation_start=date(2023, 1, 2),
        formation_end=date(2023, 12, 29),
        hedge_ratio=hedge_ratio,
        intercept=5.0,
        spread_mean=0.0,
        spread_std=2.0,
        half_life=10.0,
        coint_pvalue=0.01,
        adf_pvalue=0.005,
        hurst=0.35,
    )
    defaults.update(overrides)
    return QualifiedPair(**defaults)  # type: ignore[arg-type]


def _make_rebalancer(
    dollars_per_leg: float = 1500.0,
    max_gross: float = 3000.0,
) -> InventoryRebalancer:
    cfg = SizingConfig(dollars_per_leg=dollars_per_leg, max_gross_per_pair=max_gross)
    return InventoryRebalancer(cfg)


def _make_pos(
    pair_key: str = "AAA/BBB",
    direction: PositionDirection = PositionDirection.LONG,
    signed_qty_y: int = 30,
    signed_qty_x: int = -45,
    pair_id: int = 1,
) -> OpenPositionView:
    return OpenPositionView(
        pair_key=pair_key,
        direction=direction,
        signed_qty_y=signed_qty_y,
        signed_qty_x=signed_qty_x,
        pair_id=pair_id,
    )


# ---------------------------------------------------------------------------
# beta_target — unit tests for the core sizing math
# ---------------------------------------------------------------------------


class TestBetaTarget:
    def test_basic_beta_target(self) -> None:
        """β-weighted target: Y at dollars_per_leg, X = floor(β × qty_y)."""
        rb = _make_rebalancer(dollars_per_leg=1500, max_gross=5000)
        qty_y, qty_x = rb.beta_target(mid_y=50.0, mid_x=33.0, beta=1.5)
        assert qty_y == 30  # floor(1500/50)
        assert qty_x == 45  # floor(1.5 * 30)

    def test_capital_constraint_scales_down(self) -> None:
        """When gross exceeds max, both legs scale down."""
        rb = _make_rebalancer(dollars_per_leg=1500, max_gross=3000)
        qty_y, qty_x = rb.beta_target(mid_y=50.0, mid_x=33.0, beta=1.5)
        gross = qty_y * 50.0 + qty_x * 33.0
        assert gross <= 3000.0

    def test_x_derived_from_y_after_scaling(self) -> None:
        """After scaling, X is re-derived from Y via β (not scaled independently)."""
        rb = _make_rebalancer(dollars_per_leg=1500, max_gross=2000)
        qty_y, qty_x = rb.beta_target(mid_y=50.0, mid_x=33.0, beta=1.5)
        # X should be floor(β × qty_y), not floor(original_x × scale)
        assert qty_x == math.floor(1.5 * qty_y)

    def test_zero_price_returns_zero(self) -> None:
        """Zero mid prices produce zero quantities."""
        rb = _make_rebalancer()
        assert rb.beta_target(0.0, 50.0, 1.5) == (0, 0)
        assert rb.beta_target(50.0, 0.0, 1.5) == (0, 0)


# ---------------------------------------------------------------------------
# Reconcile — rollover scenarios
# ---------------------------------------------------------------------------


class TestReconcileRollover:
    def test_beta_unchanged_prices_unchanged_no_orders(self) -> None:
        """Same β, same prices → NO_CHANGE with zero orders."""
        rb = _make_rebalancer(dollars_per_leg=1500, max_gross=5000)
        old_pair = _make_pair(hedge_ratio=1.5)
        new_pair = _make_pair(hedge_ratio=1.5)
        pos = _make_pos(signed_qty_y=30, signed_qty_x=-45)

        results = rb.reconcile(
            {"AAA/BBB": pos},
            {"AAA/BBB": old_pair},
            {"AAA/BBB": new_pair},
            {"AAA": 50.0, "BBB": 33.0},
        )

        assert len(results) == 1
        assert results[0].action == RebalanceAction.NO_CHANGE
        assert results[0].orders == []
        assert results[0].shares_traded == 0

    def test_beta_shift_produces_delta_orders(self) -> None:
        """β shift (1.5 → 1.6) produces marginal delta orders."""
        rb = _make_rebalancer(dollars_per_leg=1500, max_gross=5000)
        old_pair = _make_pair(hedge_ratio=1.5)
        new_pair = _make_pair(hedge_ratio=1.6)

        # Old inventory: long 30 Y, short 45 X (β=1.5 × 30=45)
        pos = _make_pos(signed_qty_y=30, signed_qty_x=-45)

        results = rb.reconcile(
            {"AAA/BBB": pos},
            {"AAA/BBB": old_pair},
            {"AAA/BBB": new_pair},
            {"AAA": 50.0, "BBB": 33.0},
        )

        assert len(results) == 1
        rb_result = results[0]
        assert rb_result.action == RebalanceAction.ROLLOVER

        # Target: qty_y=30, qty_x=floor(1.6*30)=48
        # Delta: tgt_y(30) - cur_y(30)=0, tgt_x(-48) - cur_x(-45)=-3
        assert rb_result.delta_qty_y == 0
        assert rb_result.delta_qty_x == -3  # sell 3 more X (increase short)
        assert rb_result.shares_traded == 3

        # Should have exactly one order (only X delta)
        assert len(rb_result.orders) == 1
        assert rb_result.orders[0].symbol == "BBB"
        assert rb_result.orders[0].side == OrderSide.SELL
        assert rb_result.orders[0].quantity == 3

    def test_price_drift_with_capital_constraint(self) -> None:
        """Prices rallied → capital constraint forces trimming."""
        rb = _make_rebalancer(dollars_per_leg=1500, max_gross=3000)
        old_pair = _make_pair(hedge_ratio=1.5)
        new_pair = _make_pair(hedge_ratio=1.6)

        # Old position at old prices
        pos = _make_pos(signed_qty_y=30, signed_qty_x=-45)

        # Prices rallied +20%
        prices = {"AAA": 60.0, "BBB": 40.0}
        results = rb.reconcile(
            {"AAA/BBB": pos},
            {"AAA/BBB": old_pair},
            {"AAA/BBB": new_pair},
            prices,
        )

        rb_result = results[0]
        assert rb_result.action == RebalanceAction.ROLLOVER

        # Target quantities should respect capital constraint
        tgt_abs_y = abs(pos.signed_qty_y + rb_result.delta_qty_y)
        tgt_abs_x = abs(pos.signed_qty_x + rb_result.delta_qty_x)
        gross = tgt_abs_y * 60.0 + tgt_abs_x * 40.0
        assert gross <= 3000.0

        # Should be trimming (selling Y and/or buying back X)
        assert rb_result.shares_traded > 0

    def test_direction_preserved_on_rollover(self) -> None:
        """LONG direction stays LONG after rollover."""
        rb = _make_rebalancer(dollars_per_leg=1500, max_gross=5000)
        old_pair = _make_pair(hedge_ratio=1.5)
        new_pair = _make_pair(hedge_ratio=1.6)
        pos = _make_pos(
            direction=PositionDirection.LONG,
            signed_qty_y=30,
            signed_qty_x=-45,
        )

        results = rb.reconcile(
            {"AAA/BBB": pos},
            {"AAA/BBB": old_pair},
            {"AAA/BBB": new_pair},
            {"AAA": 50.0, "BBB": 33.0},
        )

        rb_result = results[0]
        # After applying delta, Y should still be positive (long)
        new_y = pos.signed_qty_y + rb_result.delta_qty_y
        new_x = pos.signed_qty_x + rb_result.delta_qty_x
        assert new_y > 0, "Y should remain long"
        assert new_x < 0, "X should remain short"

    def test_short_spread_rollover(self) -> None:
        """SHORT_SPREAD with β shift produces correct delta signs."""
        rb = _make_rebalancer(dollars_per_leg=1500, max_gross=5000)
        old_pair = _make_pair(hedge_ratio=1.5)
        new_pair = _make_pair(hedge_ratio=1.6)

        # Short spread: short Y, long X
        pos = _make_pos(
            direction=PositionDirection.SHORT,
            signed_qty_y=-30,
            signed_qty_x=45,
        )

        results = rb.reconcile(
            {"AAA/BBB": pos},
            {"AAA/BBB": old_pair},
            {"AAA/BBB": new_pair},
            {"AAA": 50.0, "BBB": 33.0},
        )

        rb_result = results[0]
        # After delta, Y should still be negative, X positive
        new_y = pos.signed_qty_y + rb_result.delta_qty_y
        new_x = pos.signed_qty_x + rb_result.delta_qty_x
        assert new_y < 0, "Y should remain short"
        assert new_x > 0, "X should remain long"

    def test_shares_traded_much_less_than_naive(self) -> None:
        """Marginal rebalancing trades far fewer shares than close + reopen."""
        rb = _make_rebalancer(dollars_per_leg=1500, max_gross=5000)
        old_pair = _make_pair(hedge_ratio=1.5)
        new_pair = _make_pair(hedge_ratio=1.6)

        pos = _make_pos(signed_qty_y=30, signed_qty_x=-45)

        results = rb.reconcile(
            {"AAA/BBB": pos},
            {"AAA/BBB": old_pair},
            {"AAA/BBB": new_pair},
            {"AAA": 50.0, "BBB": 33.0},
        )

        marginal_shares = results[0].shares_traded

        # Naive close/reopen: close sells 30 + buys 45 = 75; reopen buys 30 + sells 48 = 78
        # Total naive = 75 + 78 = 153
        naive_close = abs(pos.signed_qty_y) + abs(pos.signed_qty_x)
        # New target at β=1.6: qty_y=30, qty_x=48
        naive_open = 30 + 48
        naive_shares = naive_close + naive_open

        assert marginal_shares < naive_shares * 0.25, (
            f"Marginal ({marginal_shares}) should be <25% of "
            f"naive ({naive_shares})"
        )

    def test_beta_fidelity_after_rollover(self) -> None:
        """After rollover, qty_x ≈ β_new × qty_y (within floor rounding)."""
        rb = _make_rebalancer(dollars_per_leg=1500, max_gross=5000)
        old_pair = _make_pair(hedge_ratio=1.5)
        new_pair = _make_pair(hedge_ratio=1.6)
        pos = _make_pos(signed_qty_y=30, signed_qty_x=-45)

        results = rb.reconcile(
            {"AAA/BBB": pos},
            {"AAA/BBB": old_pair},
            {"AAA/BBB": new_pair},
            {"AAA": 50.0, "BBB": 33.0},
        )

        rb_result = results[0]
        new_abs_y = abs(pos.signed_qty_y + rb_result.delta_qty_y)
        new_abs_x = abs(pos.signed_qty_x + rb_result.delta_qty_x)
        expected_x = math.floor(1.6 * new_abs_y)
        assert new_abs_x == expected_x, (
            f"Post-rollover X={new_abs_x}, expected floor(1.6×{new_abs_y})={expected_x}"
        )


# ---------------------------------------------------------------------------
# Reconcile — force-exit scenarios
# ---------------------------------------------------------------------------


class TestReconcileForceExit:
    def test_dropped_pair_force_exited(self) -> None:
        """Pair not in new_pairs → FORCE_EXIT with full liquidation orders."""
        rb = _make_rebalancer()
        old_pair = _make_pair(hedge_ratio=1.5)
        pos = _make_pos(signed_qty_y=30, signed_qty_x=-45)

        results = rb.reconcile(
            {"AAA/BBB": pos},
            {"AAA/BBB": old_pair},
            {},  # pair didn't re-qualify
            {"AAA": 50.0, "BBB": 33.0},
        )

        assert len(results) == 1
        assert results[0].action == RebalanceAction.FORCE_EXIT
        assert len(results[0].orders) == 2

        # Should liquidate: SELL 30 Y (close long), BUY 45 X (cover short)
        y_order = results[0].orders[0]
        x_order = results[0].orders[1]
        assert y_order.symbol == "AAA"
        assert y_order.side == OrderSide.SELL
        assert y_order.quantity == 30
        assert x_order.symbol == "BBB"
        assert x_order.side == OrderSide.BUY
        assert x_order.quantity == 45

    def test_force_exit_delta_is_negative_inventory(self) -> None:
        """Force-exit delta should zero out the position."""
        rb = _make_rebalancer()
        pos = _make_pos(signed_qty_y=30, signed_qty_x=-45)

        results = rb.reconcile(
            {"AAA/BBB": pos}, {}, {}, {"AAA": 50.0, "BBB": 33.0},
        )

        assert results[0].delta_qty_y == -30  # cancel +30
        assert results[0].delta_qty_x == 45   # cancel -45

    def test_force_exit_short_spread(self) -> None:
        """Force-exit a SHORT_SPREAD position."""
        rb = _make_rebalancer()
        pos = _make_pos(
            direction=PositionDirection.SHORT,
            signed_qty_y=-20,
            signed_qty_x=30,
        )

        results = rb.reconcile(
            {"AAA/BBB": pos}, {}, {}, {"AAA": 50.0, "BBB": 33.0},
        )

        assert results[0].action == RebalanceAction.FORCE_EXIT
        # Should BUY 20 Y (cover short) and SELL 30 X (close long)
        y_order = results[0].orders[0]
        x_order = results[0].orders[1]
        assert y_order.side == OrderSide.BUY
        assert y_order.quantity == 20
        assert x_order.side == OrderSide.SELL
        assert x_order.quantity == 30


# ---------------------------------------------------------------------------
# Reconcile — mixed scenarios
# ---------------------------------------------------------------------------


class TestReconcileMixed:
    def test_multiple_pairs_mixed_actions(self) -> None:
        """Mix of rollover, force-exit, and no-change in one reconciliation."""
        rb = _make_rebalancer(dollars_per_leg=1500, max_gross=5000)

        pair_a_old = _make_pair(symbol_y="A1", symbol_x="A2", hedge_ratio=1.5)
        pair_b_old = _make_pair(symbol_y="B1", symbol_x="B2", hedge_ratio=2.0)
        pair_c_old = _make_pair(symbol_y="C1", symbol_x="C2", hedge_ratio=1.0)

        # Only A and C re-qualify (B dropped); A has new β, C unchanged
        pair_a_new = _make_pair(symbol_y="A1", symbol_x="A2", hedge_ratio=1.6)
        pair_c_new = _make_pair(symbol_y="C1", symbol_x="C2", hedge_ratio=1.0)

        active = {
            "A1/A2": _make_pos(
                pair_key="A1/A2", signed_qty_y=30, signed_qty_x=-45, pair_id=1,
            ),
            "B1/B2": _make_pos(
                pair_key="B1/B2", signed_qty_y=15, signed_qty_x=-30, pair_id=2,
            ),
            "C1/C2": _make_pos(
                pair_key="C1/C2", signed_qty_y=30, signed_qty_x=-30, pair_id=3,
            ),
        }

        old_pairs = {"A1/A2": pair_a_old, "B1/B2": pair_b_old, "C1/C2": pair_c_old}
        new_pairs = {"A1/A2": pair_a_new, "C1/C2": pair_c_new}
        prices = {
            "A1": 50.0, "A2": 33.0,
            "B1": 100.0, "B2": 50.0,
            "C1": 50.0, "C2": 50.0,
        }

        results = rb.reconcile(active, old_pairs, new_pairs, prices)
        actions = {r.pair_key: r.action for r in results}

        assert actions["A1/A2"] == RebalanceAction.ROLLOVER
        assert actions["B1/B2"] == RebalanceAction.FORCE_EXIT
        assert actions["C1/C2"] == RebalanceAction.NO_CHANGE

    def test_missing_prices_returns_no_change(self) -> None:
        """Missing mid prices for a rollover pair → NO_CHANGE (skip)."""
        rb = _make_rebalancer()
        old_pair = _make_pair(hedge_ratio=1.5)
        new_pair = _make_pair(hedge_ratio=1.6)
        pos = _make_pos(signed_qty_y=30, signed_qty_x=-45)

        results = rb.reconcile(
            {"AAA/BBB": pos},
            {"AAA/BBB": old_pair},
            {"AAA/BBB": new_pair},
            {"AAA": 50.0},  # BBB price missing
        )

        assert results[0].action == RebalanceAction.NO_CHANGE

    def test_rebalance_result_immutable(self) -> None:
        """RebalanceResult is a frozen dataclass."""
        result = RebalanceResult(
            pair_key="AAA/BBB",
            action=RebalanceAction.NO_CHANGE,
            orders=[],
            old_beta=1.5,
            new_beta=1.5,
            delta_qty_y=0,
            delta_qty_x=0,
            shares_traded=0,
        )
        with pytest.raises(AttributeError):
            result.action = RebalanceAction.ROLLOVER  # type: ignore[misc]

    def test_orders_are_not_entry(self) -> None:
        """All rebalance/force-exit orders have is_entry=False."""
        rb = _make_rebalancer(dollars_per_leg=1500, max_gross=5000)
        old_pair = _make_pair(hedge_ratio=1.5)
        new_pair = _make_pair(hedge_ratio=1.6)
        pos = _make_pos(signed_qty_y=30, signed_qty_x=-45)

        results = rb.reconcile(
            {"AAA/BBB": pos},
            {"AAA/BBB": old_pair},
            {"AAA/BBB": new_pair},
            {"AAA": 50.0, "BBB": 33.0},
        )

        for order in results[0].orders:
            assert order.is_entry is False

    def test_empty_active_positions(self) -> None:
        """No active positions → empty results."""
        rb = _make_rebalancer()
        results = rb.reconcile({}, {}, {}, {})
        assert results == []
