"""
Custom tyssue effectors.

    LateralAreaElasticity: elasticity of the lateral area h P of each cell (effective 2D model)
    TiltedLateralAreaElasticity: same, with basal vertices anchored, so lateral faces tilt
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


def basal_offsets(eptm):
    """
    Horizontal offset of each apical vertex from its basal vertex, apical - basal (minimum image
    on periodic sheets). Zero until the basal vertices are anchored (settings['basal_anchored'])
    """
    offsets = np.zeros((eptm.Nv, 2))
    if eptm.settings.get("basal_anchored", False):
        for i, coord in enumerate(("x", "y")):
            offsets[:, i] = eptm.vert_df[coord] - eptm.vert_df["basal_" + coord]
            boundary = (eptm.settings.get("boundaries") or {}).get(coord)
            if boundary is not None:
                period = boundary[1] - boundary[0]
                offsets[:, i] -= period * np.round(offsets[:, i] / period)
    return offsets


def tilted_lateral_faces(eptm):
    """
    Lateral face of each half-edge: the quadrilateral between the apical edge (s -> t, at the
    cell height h) and the basal edge (anchored vertices, at height 0). Its area is half the
    norm of the cross product of its diagonals:
        D1 = basal_t - apical_s = (e - d_t, -h),   D2 = apical_t - basal_s = (e + d_s, h)
    with e the apical edge vector and d the apical - basal offsets.

    The lateral area of the effective cell is h |e| (its polygon stands for both apical and basal
    surfaces) plus the extra area due to the tilt, Q - h (|e| + |e_basal|) / 2 (tilted face minus
    the same face upright). That is Q + h (|e| - |e_basal|) / 2, which equals h |e| without tilt,
    with the same forces, so anchoring the basal vertices does not move a relaxed tissue
    :return: D1, D2 (Ne x 3), cross product D1 x D2 (Ne x 3) and lateral areas per half-edge (Ne)
    """
    edges = eptm.edge_df
    offsets = basal_offsets(eptm)
    srcePosition = eptm.vert_df.index.get_indexer(edges["srce"])
    trgtPosition = eptm.vert_df.index.get_indexer(edges["trgt"])
    height = eptm.upcast_face(eptm.face_df["height"]).to_numpy()
    edgeVector = edges[["dx", "dy"]].to_numpy()

    diagonal1 = np.column_stack([edgeVector - offsets[trgtPosition], -height])
    diagonal2 = np.column_stack([edgeVector + offsets[srcePosition], height])
    cross = np.cross(diagonal1, diagonal2)
    tiltedFaceArea = 0.5 * np.linalg.norm(cross, axis=1)

    basalEdgeLength = np.linalg.norm(edgeVector - offsets[trgtPosition] + offsets[srcePosition], axis=1)
    apicalEdgeLength = np.linalg.norm(edgeVector, axis=1)
    return diagonal1, diagonal2, cross, tiltedFaceArea + 0.5 * height * (apicalEdgeLength - basalEdgeLength)


class TiltedLateralAreaElasticity(AbstractEffector):
    """
    Elasticity of the lateral area of each cell when its lateral faces can tilt: the apical
    vertices move, the basal ones stay where they were anchored (vert_df 'basal_x', 'basal_y',
    set at the ablation). The lateral area grows when the apical vertices move away from their
    basal ones, so the recoil of the apical surface is resisted by the tilted lateral faces:
        E = 1/2 lateral_elasticity (h P + tilt extra area - prefered_lateral_area)^2
    (see tilted_lateral_faces)
    Before the anchoring, it is the same as LateralAreaElasticity. Only in-plane forces: use it
    with a fixed height
    """

    label = "Tilted lateral area elasticity"
    element = "face"
    magnitude = "lateral_elasticity"
    specs = {
        "face": {"is_alive": 1, "lateral_elasticity": 1.0, "prefered_lateral_area": 1.0},
        "vert": {"basal_x": 0.0, "basal_y": 0.0},
        "settings": {"basal_anchored": False},
    }

    @staticmethod
    def lateral_area(eptm, faceAreas=None):
        """Lateral area of each cell: sum of the areas of its lateral faces"""
        if faceAreas is None:
            _, _, _, faceAreas = tilted_lateral_faces(eptm)
        return pd.Series(faceAreas, index=eptm.edge_df.index).groupby(eptm.edge_df["face"]).sum()

    @staticmethod
    def energy(eptm):
        if not eptm.settings.get("basal_anchored", False):
            return LateralAreaElasticity.energy(eptm)
        df = eptm.face_df
        lateralArea = TiltedLateralAreaElasticity.lateral_area(eptm)
        return 0.5 * df["is_alive"] * df["lateral_elasticity"] * (lateralArea - df["prefered_lateral_area"]) ** 2

    @staticmethod
    def gradient(eptm):
        if not eptm.settings.get("basal_anchored", False):
            return LateralAreaElasticity.gradient(eptm)
        df = eptm.face_df
        diagonal1, diagonal2, cross, faceAreas = tilted_lateral_faces(eptm)
        lateralArea = TiltedLateralAreaElasticity.lateral_area(eptm, faceAreas)
        dE_dLateral = eptm.upcast_face(
            df["is_alive"] * df["lateral_elasticity"] * (lateralArea - df["prefered_lateral_area"])).to_numpy()

        # Tilted face: d|D1 x D2| = dD1 . (D2 x n) + dD2 . (n x D1). With anchored basal vertices,
        # D1 = basal_t - apical_s only depends on the source and D2 = apical_t - basal_s on the target
        normal = cross / np.linalg.norm(cross, axis=1)[:, None]
        grad_srce = -0.5 * np.cross(diagonal2, normal)
        grad_trgt = 0.5 * np.cross(normal, diagonal1)
        # + h/2 |e|: the apical edge length pulls along the edge (the basal edge is fixed)
        halfHeight = 0.5 * eptm.upcast_face(df["height"]).to_numpy()[:, None]
        unit = np.column_stack([eptm.edge_df[["ux", "uy"]].to_numpy(), np.zeros(eptm.Ne)])
        grad_srce = (grad_srce - halfHeight * unit) * dE_dLateral[:, None]
        grad_trgt = (grad_trgt + halfHeight * unit) * dE_dLateral[:, None]
        grad_srce[:, 2] = 0.0  # fixed height
        grad_trgt[:, 2] = 0.0

        columns = ["g" + c for c in eptm.coords]
        return (pd.DataFrame(grad_srce, index=eptm.edge_df.index, columns=columns),
                pd.DataFrame(grad_trgt, index=eptm.edge_df.index, columns=columns))


def anchor_basal_vertices(eptm):
    """Fix the basal vertices under the current apical ones: from now on, the lateral faces tilt"""
    eptm.vert_df["basal_x"] = eptm.vert_df["x"]
    eptm.vert_df["basal_y"] = eptm.vert_df["y"]
    eptm.settings["basal_anchored"] = True


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
