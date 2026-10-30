"""
Custom tyssue effectors.

    LateralAreaElasticity: elasticity of the lateral area h P of each cell (effective 2D model)
    WoundContractility: purse string on the wound edges (settings['wound_tension'])
"""
import numpy as np
import pandas as pd
from tyssue.dynamics.effectors import AbstractEffector
from tyssue.utils import to_nd


def lateral_area(eptm):
    """Lateral area of each cell: height * perimeter (face_df 'height' is the mean vertex height)"""
    return eptm.face_df["height"] * eptm.face_df["perimeter"]


class LateralAreaElasticity(AbstractEffector):
    """
    Elasticity of the lateral area of each cell, A_lat = h P (2.5D sheet, SheetGeometry):
        E = 1/2 lateral_elasticity (A_lat - prefered_lateral_area)^2
    In the plane, dA_lat = h dP, so the cell pulls its edges with the tension
    lateral_elasticity (A_lat - prefered_lateral_area) h. Along z (free height), dA_lat = P dh,
    and h is the mean height of the cell vertices
    """

    label = "Lateral area elasticity"
    element = "face"
    magnitude = "lateral_elasticity"
    specs = {"face": {"is_alive": 1, "lateral_elasticity": 1.0, "prefered_lateral_area": 1.0}}

    @staticmethod
    def energy(eptm):
        df = eptm.face_df
        return 0.5 * df["is_alive"] * df["lateral_elasticity"] * (lateral_area(eptm) - df["prefered_lateral_area"]) ** 2

    @staticmethod
    def gradient(eptm):
        df = eptm.face_df
        dE_dLateral = df["is_alive"] * df["lateral_elasticity"] * (lateral_area(eptm) - df["prefered_lateral_area"])

        # In plane: dA_lat/dP = h, and the perimeter pulls along each edge
        tension = to_nd(eptm.upcast_face(dE_dLateral * df["height"]), eptm.dim)
        unit = eptm.edge_df[eptm.ucoords].to_numpy()
        grad_srce = pd.DataFrame(-unit * tension, index=eptm.edge_df.index, columns=["g" + c for c in eptm.coords])

        # Along the height axis: dA_lat/dh_v = P / num_sides (h is the mean of the vertex heights)
        height_axis = "g" + eptm.settings.get("height_axis", "z")
        grad_height = eptm.upcast_face(dE_dLateral * df["perimeter"] / df["num_sides"]).to_numpy()
        grad_trgt = -grad_srce.copy()
        grad_srce[height_axis] += grad_height
        return grad_srce, grad_trgt


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
        k = to_nd(wound_edge_tension(eptm), eptm.dim)
        u = eptm.edge_df[eptm.ucoords].to_numpy()
        g_s = pd.DataFrame(-u * k, index=eptm.edge_df.index, columns=["g" + c for c in eptm.coords])
        return g_s, -g_s


EFFECTORS = [WoundContractility]
