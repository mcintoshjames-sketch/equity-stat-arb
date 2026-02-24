"""Symbol universe and sector mapping."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stat_arb.config.settings import UniverseConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Universe:
    """Holds the symbol universe with sector mappings."""

    symbols: list[str]
    sector_map: dict[str, str]  # symbol → sector
    sector_symbols: dict[str, list[str]]  # sector → [symbols]

    @property
    def sector_pairs(self) -> list[tuple[str, str, str]]:
        """Generate all intra-sector pairs as (sym_a, sym_b, sector)."""
        pairs = []
        for sector, syms in self.sector_symbols.items():
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    pairs.append((syms[i], syms[j], sector))
        return pairs


def load_universe(config: UniverseConfig) -> Universe:
    """Build Universe from UniverseConfig sectors dict."""
    symbols: list[str] = []
    sector_map: dict[str, str] = {}
    sector_symbols: dict[str, list[str]] = {}

    for sector, syms in config.sectors.items():
        sector_symbols[sector] = list(syms)
        for sym in syms:
            if sym not in sector_map:
                symbols.append(sym)
                sector_map[sym] = sector
            else:
                logger.warning("Duplicate symbol %s in sector %s (already in %s)",
                               sym, sector, sector_map[sym])

    logger.info("Loaded universe: %d symbols across %d sectors",
                len(symbols), len(sector_symbols))
    return Universe(symbols=symbols, sector_map=sector_map, sector_symbols=sector_symbols)
