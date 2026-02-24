"""Symbol universe and sector mapping.

The :class:`Universe` is a frozen dataclass built from ``UniverseConfig``
that provides the tradable symbol pool and all intra-sector pair
combinations for the discovery pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stat_arb.config.settings import UniverseConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Universe:
    """Immutable snapshot of the tradable symbol universe.

    Attributes:
        symbols: Flat list of all unique symbols across sectors.
        sector_map: Mapping of symbol → sector name for quick lookup.
        sector_symbols: Mapping of sector name → list of symbols in that sector.
    """

    symbols: list[str]
    sector_map: dict[str, str]
    sector_symbols: dict[str, list[str]]

    @property
    def sector_pairs(self) -> list[tuple[str, str, str]]:
        """Generate all intra-sector pairs as ``(sym_a, sym_b, sector)``.

        Pairs are generated in sorted order within each sector (i < j)
        to ensure deterministic, non-duplicate combinations.
        """
        pairs = []
        for sector, syms in self.sector_symbols.items():
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    pairs.append((syms[i], syms[j], sector))
        return pairs


def load_universe(config: UniverseConfig) -> Universe:
    """Build a :class:`Universe` from a ``UniverseConfig`` sectors dict.

    Deduplicates symbols that appear in multiple sectors (keeping the first
    occurrence) and logs a warning for each duplicate.

    Args:
        config: Universe configuration subsection.

    Returns:
        Frozen ``Universe`` instance ready for pair generation.
    """
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
                logger.warning(
                    "Duplicate symbol %s in sector %s (already in %s)",
                    sym, sector, sector_map[sym],
                )

    logger.info(
        "Loaded universe: %d symbols across %d sectors",
        len(symbols), len(sector_symbols),
    )
    return Universe(symbols=symbols, sector_map=sector_map, sector_symbols=sector_symbols)
