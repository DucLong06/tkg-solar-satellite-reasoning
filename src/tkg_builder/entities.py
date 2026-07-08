"""8 entity types of the temporal knowledge graph (paper §3 / pland Part 6).

MVP design (documented in docs/assumptions.md): one node per entity TYPE (N=8),
and the graph is built PER SAMPLE — each node carries the current sample's pooled
features plus a learned type embedding. This fixes the batch<->node correspondence
(design note): the graph learner returns one [B, 128] embedding per batch sample,
not a single per-node tensor broadcast across the batch.
"""

from __future__ import annotations

from enum import IntEnum


class EntityType(IntEnum):
    PV_PLANT = 0
    CLOUD_REGION = 1
    METEO_STATION = 2
    GEO_REGION = 3
    IOT_SENSOR = 4
    WEATHER_STATE = 5
    IRRADIANCE_INFO = 6
    LOAD_STATE = 7


N_ENTITY_TYPES = len(EntityType)  # 8


# Fixed, synthetic node coordinates (lat, lon) within a single ROI so the geo
# term is at least structurally computable. NOTE: with the
# faithful-to-paper mismatched data (Euro PV vs Asia meteo/sat) the geo edge is
# NOT physically meaningful — kept for mechanical fidelity only, see assumptions.
NODE_COORDS: tuple[tuple[float, float], ...] = (
    (10.80, 106.65),  # PV plant
    (10.90, 106.70),  # cloud region
    (10.82, 106.63),  # meteo station
    (10.85, 106.60),  # geo region
    (10.78, 106.68),  # IoT sensor
    (10.88, 106.66),  # weather state
    (10.83, 106.64),  # irradiance info
    (10.81, 106.69),  # load state
)
