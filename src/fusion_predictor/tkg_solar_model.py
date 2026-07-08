"""Top-level proposed model: TKGSolarModel (paper §3.6 fusion).

Wires the 3 encoders (meteo, satellite, graph learner) into the fusion
fusion predictor. forward(batch) -> [B, N_HORIZONS], matching the model interface
the training loop expects.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.common.shapes import EMBED_DIM
from src.meteo_encoder.meteo_encoder import MeteoEncoder
from src.satellite_encoder.satellite_encoder import SatelliteEncoder
from src.graph_learner.graph_learner import GraphLearner
from src.fusion_predictor.fusion_predictor import FusionPredictor


class TKGSolarModel(nn.Module):
    def __init__(
        self,
        sat_encoder: SatelliteEncoder,
        meteo_encoder: MeteoEncoder,
        graph_learner: GraphLearner,
        fusion: FusionPredictor,
        use_sat: bool = True,
        use_meteo: bool = True,
        use_graph: bool = True,
    ) -> None:
        super().__init__()
        self.sat_encoder = sat_encoder
        self.meteo_encoder = meteo_encoder
        self.graph_learner = graph_learner
        self.fusion = fusion
        # Ablation switches: a disabled branch contributes a zero embedding (keeps
        # the fusion concat shape, removes the branch's signal). All-on = full model.
        self.use_sat = use_sat
        self.use_meteo = use_meteo
        self.use_graph = use_graph

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        f_sat = self.sat_encoder(batch["sat_seq"]) if self.use_sat else None       # [B,128]
        h_met = self.meteo_encoder(batch["meteo_seq"]) if self.use_meteo else None  # [B,128]
        h_graph = (
            self.graph_learner(batch["meteo_seq"], batch["pv_hist"])
            if self.use_graph else None
        )                                                                           # [B,128]
        ref = next(t for t in (f_sat, h_met, h_graph) if t is not None)
        zero = torch.zeros(ref.shape[0], EMBED_DIM, device=ref.device, dtype=ref.dtype)
        return self.fusion(
            f_sat if f_sat is not None else zero,
            h_met if h_met is not None else zero,
            h_graph if h_graph is not None else zero,
        )                                                                           # [B,3]

    @classmethod
    def from_config(cls, cfg) -> "TKGSolarModel":
        sat = SatelliteEncoder(
            out_dim=EMBED_DIM, backbone=cfg.sat_backbone,
            pretrained=cfg.pretrained_backbone, freeze_backbone=cfg.freeze_backbone,
            n_heads=cfg.n_heads, dropout=cfg.dropout,
        )
        meteo = MeteoEncoder(hidden_dim=EMBED_DIM, out_dim=EMBED_DIM, n_heads=cfg.n_heads, dropout=cfg.dropout)
        graph = GraphLearner(
            gat_dim=EMBED_DIM, out_dim=EMBED_DIM, n_nodes=cfg.n_graph_nodes,
            n_heads=cfg.n_heads, dropout=cfg.dropout,
        )
        fusion = FusionPredictor(hidden_dim=EMBED_DIM * 2, dropout=cfg.dropout)
        return cls(
            sat, meteo, graph, fusion,
            use_sat=getattr(cfg, "use_sat", True),
            use_meteo=getattr(cfg, "use_meteo", True),
            use_graph=getattr(cfg, "use_graph", True),
        )
