"""
tyssue effectors of the effective 2D model (see params.py for the derivation).

Face columns used (set by tissue.set_reference_state):
    is_alive, area, perimeter
    lambdaV, area_v0                       -> EffectiveVolume
    surf_alpha = lS1 + lS3                 -> EffectiveSurface
    surf_beta  = lS2 * h
    surf_c0    = ref_A0 (alpha A_i + beta P_i)
    surf_S0    = ref_A0 (2 A_i + h P_i)    (Cell.Area0 in 3D)
Settings:
    wound_tension                          -> WoundContractility, applied to the wound edges
"""
import numpy as np
import pandas as pd
from tyssue.dynamics.effectors import AbstractEffector
from tyssue.dynamics.planar_gradients import area_grad as area_grad2d
from tyssue.utils import to_nd


def wound_edge_tension(eptm):
    """Per-edge tension: settings['wound_tension'] on the half-edges of alive cells facing
    the debris, 0 elsewhere. Recomputed at every call, so it stays right after T1s."""
    T = eptm.settings.get("wound_tension", 0.0)
    e = eptm.edge_df
    if T == 0 or "is_debris" not in eptm.face_df or not eptm.face_df["is_debris"].any():
        return pd.Series(0.0, index=e.index)
    debris = eptm.face_df["is_debris"].reindex(e["face"]).to_numpy()
    opp = e["opposite"].to_numpy()
    opp_face = np.where(opp >= 0, e["face"].reindex(opp).to_numpy(), -1)
    opp_debris = np.zeros(len(e), dtype=bool)
    ok = opp_face >= 0
    opp_debris[ok] = eptm.face_df["is_debris"].reindex(opp_face[ok].astype(int)).to_numpy() == 1
    return pd.Series(np.where(opp_debris & (debris == 0), T, 0.0), index=e.index)


class WoundContractility(AbstractEffector):
    """KgContractility on the wound edge: energy = wound_tension * length."""

    label = "Wound contractility (purse string + lateral cables)"
    element = "edge"
    magnitude = "wound_tension"
    specs = {"settings": {"wound_tension": 0.0}}

    @staticmethod
    def energy(eptm):
        return wound_edge_tension(eptm) * eptm.edge_df["length"]

    @staticmethod
    def gradient(eptm):
        k = to_nd(wound_edge_tension(eptm), 2)
        u = eptm.edge_df[["ux", "uy"]].to_numpy()
        g_s = pd.DataFrame(-u * k, index=eptm.edge_df.index, columns=["gx", "gy"])
        return g_s, -g_s


EFFECTORS = [WoundContractility]
