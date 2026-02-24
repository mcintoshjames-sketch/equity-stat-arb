"""Tests for universe loading."""

from stat_arb.data.universe import load_universe


class TestUniverse:
    def test_load_universe(self, universe_config):
        u = load_universe(universe_config)
        assert len(u.symbols) == 6
        assert u.sector_map["AAPL"] == "technology"
        assert u.sector_map["JPM"] == "financials"

    def test_sector_pairs(self, universe_config):
        u = load_universe(universe_config)
        pairs = u.sector_pairs
        # C(3,2) per sector = 3, two sectors = 6
        assert len(pairs) == 6
        # Each pair has (sym_a, sym_b, sector)
        sectors = {p[2] for p in pairs}
        assert sectors == {"technology", "financials"}
