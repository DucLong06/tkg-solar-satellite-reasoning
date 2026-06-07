"""7 relation types of the temporal knowledge graph (paper §3 / pland Part 6).

These name the ``r`` of the quadruple (h, r, o, t). In the MVP the adjacency is
built from geo + feature similarity (see adjacency_builder); relation-typed edge
modulation is deferred (documented assumption A6) — the enum is kept so the graph
semantics are explicit and relation typing can be wired in later.
"""

from __future__ import annotations

from enum import IntEnum


class RelationType(IntEnum):
    AFFECTS = 0
    CONNECTED_TO = 1
    LOCATED_IN = 2
    INFLUENCES = 3
    CAUSES = 4
    MOVES_TOWARD = 5
    CORRELATED_WITH = 6


N_RELATION_TYPES = len(RelationType)  # 7
