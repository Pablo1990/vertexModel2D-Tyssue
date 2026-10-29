import numpy as np
import pandas as pd
from tyssue import PlanarGeometry

from src.inputMechanicalParameters import wound_contractility

def central_faces(sheet, n):
    """n faces closest to the tissue centre (a connected cluster for a regular-ish sheet)."""
    b = sheet.settings.get("boundaries")
    if b:  # periodic: centre of the box, so the wound does not cross it
        c = np.array([np.mean(b["x"]), np.mean(b["y"])])
    elif "is_border" in sheet.vert_df:
        c = sheet.vert_df.loc[sheet.vert_df["is_border"] == 0, ["x", "y"]].mean().to_numpy()
    else:
        c = sheet.vert_df[["x", "y"]].mean().to_numpy()
    d = np.hypot(sheet.face_df["x"] - c[0], sheet.face_df["y"] - c[1])
    return d.sort_values().index[:n].to_numpy()


def ablate(sheet, faces, geom=PlanarGeometry):
    """Combine the ablated cells into a single debris face (Geo.ablate_cells, combine_cells=True).

    The debris has no surface energy and no contractility (is_alive = 0); its
    3D volume stiffness is lambdaV * 1e-8, i.e. negligible, so it is omitted.
    """
    faces = np.asarray(faces)
    debris = faces[0]
    if "is_debris" not in sheet.face_df:
        sheet.face_df["is_debris"] = 0
    geom.update_all(sheet)
    e = sheet.edge_df
    in_f = e["face"].isin(faces)
    opp = e["opposite"]
    opp_face = np.where(opp >= 0, e["face"].reindex(opp).to_numpy(), -1)
    internal = in_f & np.isin(opp_face, faces)
    sheet.edge_df = e[~internal].copy()
    sheet.edge_df.loc[sheet.edge_df["face"].isin(faces), "face"] = debris
    sheet.face_df = sheet.face_df.drop(faces[1:])
    used = np.unique(sheet.edge_df[["srce", "trgt"]].to_numpy())
    sheet.vert_df = sheet.vert_df.loc[used]
    sheet.face_df.loc[debris, "is_alive"] = 0
    sheet.face_df.loc[debris, "is_debris"] = 1
    sheet.reset_index(order=True)
    sheet.reset_topo()
    sheet.network_changed = True
    geom.update_all(sheet)
    # Wound area at the ablation, reference of the relative wound area
    sheet.face_df["area0"] = sheet.face_df["area"]
    return sheet


def debris_face(sheet):
    if "is_debris" not in sheet.face_df or not sheet.face_df["is_debris"].any():
        return None
    return sheet.face_df.index[sheet.face_df["is_debris"] == 1][0]


def wound_edges(sheet):
    """Half-edges of alive cells facing the debris (len(SharedByCells) > 1 with a dead cell in 3D)."""
    d = debris_face(sheet)
    if d is None:
        return sheet.edge_df.index[:0]
    e = sheet.edge_df
    opp = e["opposite"]
    opp_face = np.where(opp >= 0, e["face"].reindex(opp).to_numpy(), -1)
    debris = sheet.face_df.index[sheet.face_df["is_debris"] == 1]
    return e.index[np.isin(opp_face, debris) & ~e["face"].isin(debris)]


def update_wound_tension(sheet, t):
    sheet.settings["wound_tension"] = wound_contractility(sheet, t)


def wound_measures(sheet):
    d = debris_face(sheet)
    if d is None:
        return {"wound_area": 0.0, "wound_perimeter": 0.0, "num_cells_wound_edge": 0}
    we = wound_edges(sheet)
    return {
        "wound_area": float(abs(sheet.face_df.loc[d, "area"])),
        "wound_perimeter": float(sheet.face_df.loc[d, "perimeter"]),
        "num_cells_wound_edge": int(sheet.edge_df.loc[we, "face"].nunique()),
        "relative_wound_area": 100 * float(abs(sheet.face_df.loc[d, "area"])) / float(abs(sheet.face_df.loc[d, "area0"]))
    }


def wound_record(sheet, t):
    """Wound measures at time t, as a row of the wound time course"""
    return {"time": t, "time_after_ablation": t - sheet.settings["TInitAblation"], **wound_measures(sheet)}


def wound_features(wound_timecourse):
    """
    Features of a wound time course, used to compare and calibrate simulations
    :param wound_timecourse: DataFrame with 'time_after_ablation' and 'wound_area'
    :return: dict with
        recoil: initial expansion rate, (A(t1) - A0) / (A0 t1) at the first record (1/min)
        max_wound_area: max(A) / A0
        time_max_wound_area: time after ablation of the maximum (min)
        healing_rate: slope of A / A0 after the maximum (1/min, negative when closing)
    """
    t = wound_timecourse["time_after_ablation"].to_numpy()
    area = wound_timecourse["wound_area"].to_numpy() / wound_timecourse["wound_area"].iloc[0]
    i_max = int(np.argmax(area))
    healing_rate = np.polyfit(t[i_max:], area[i_max:], 1)[0] if len(t) - i_max > 1 else 0.0
    return {
        "recoil": float((area[1] - 1) / t[1]) if len(t) > 1 else 0.0,
        "max_wound_area": float(area[i_max]),
        "time_max_wound_area": float(t[i_max]),
        "healing_rate": float(healing_rate),
    }


def relative_wound_area(wound_timecourse, timeAfterAblation):
    """
    Relative wound area (%) at a time after ablation: 100 * A(t) / A(0), so 100% at the ablation
    :param wound_timecourse: DataFrame with 'time_after_ablation' (min) and 'wound_area'
    :param timeAfterAblation: minutes
    :return: relative wound area (%)
    """
    t = wound_timecourse["time_after_ablation"].to_numpy()
    area = wound_timecourse["wound_area"].to_numpy()
    return float(100 * np.interp(timeAfterAblation, t, area) / np.interp(0.0, t, area))


def load_wound_area_target(csvFile, timeAfterAblation):
    """
    Target relative wound area (%) from experimental data
    :param csvFile: CSV with columns 'time_after_ablation' (min) and 'wound_area' (any unit, or
        already in % of the area at t = 0)
    :param timeAfterAblation: minutes
    :return: relative wound area (%)
    """
    return relative_wound_area(pd.read_csv(csvFile), timeAfterAblation)


def edge_recoil(sheet):
    """
    Mean initial recoil velocity after an edge ablation (length per minute).

    As in Farhadifar et al. 2007, an edge shared by cells f and g carries the tension
    T = Lambda + Gamma_f P_f + Gamma_g P_g (line tension plus the perimeter contractility
    of both cells). Ablating the edge removes T; from mechanical equilibrium its two vertices
    then separate at 2 T / viscosity. The volume term acts on the cell areas, not on single
    edges, so it does not contribute.
    :param sheet: relaxed tissue
    :return: mean over the edges shared by two alive cells
    """
    faces = sheet.face_df
    contractility = faces["is_alive"] * faces["contractility"] * faces["perimeter"]
    # Each half-edge carries half of the line tension (tyssue LineTension)
    halfTension = 0.5 * sheet.edge_df["line_tension"] + contractility.reindex(sheet.edge_df["face"]).to_numpy()
    edges = sheet.edge_df[sheet.edge_df["opposite"] >= 0]
    tension = halfTension.loc[edges.index].to_numpy() + halfTension.loc[edges["opposite"]].to_numpy()
    viscosity = sheet.vert_df["viscosity"].reindex(edges["srce"]).to_numpy()
    return float(np.mean(2 * tension / viscosity))


def calibration_error(features, targets, weights=None):
    """
    Squared relative error between simulated features and target features
    :param features: dict from wound_features
    :param targets: dict with the target values of some features, e.g. {'max_wound_area': 1.65}
    :param weights: optional dict with the weight of each feature (default 1)
    :return: error (0 if all the targets are matched)
    """
    weights = weights or {}
    return sum(weights.get(name, 1.0) * ((features[name] - target) / target) ** 2
               for name, target in targets.items())
