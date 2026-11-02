import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
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


def opposite_faces(sheet):
    """Face on the other side of each half-edge (-1 for border half-edges)"""
    e = sheet.edge_df
    opp = e["opposite"].to_numpy()
    return np.where(opp >= 0, e["face"].reindex(opp).to_numpy(), -1)


def shared_half_edges(sheet, faceA, faceB):
    """
    The edge shared by faceA and faceB (Geo.get_edges_vertices). tyssue stores each edge as two
    half-edges, one per cell (same vertices, opposite directions): both are returned
    """
    e = sheet.edge_df
    faces = e["face"].to_numpy()
    opp_face = opposite_faces(sheet)
    shared = ((faces == faceA) & (opp_face == faceB)) | ((faces == faceB) & (opp_face == faceA))
    return e.index[shared]


def shared_edge_length(sheet, faceA, faceB):
    """
    Length of the edge(s) shared by faceA and faceB (Geo.get_edge_length): each edge is counted
    once, from the half-edges of faceA
    :return: 0 if they are not neighbours
    """
    e = sheet.edge_df
    shared = shared_half_edges(sheet, faceA, faceB)
    return float(e.loc[shared[e.loc[shared, "face"].to_numpy() == faceA], "length"].sum())


def ablate_edge(sheet, faceA, faceB):
    """
    Edge ablation (Geo.ablate_edge with adjacent_surface=False): the edge shared by faceA and
    faceB (both of its half-edges, one per cell) loses its own tension: the whole line tension
    and the pull of the perimeter or lateral surface of both cells along it, see
    effectors.without_ablated_edges. The cell areas/volumes still act on its vertices
    :return: the two half-edges of the ablated edge
    """
    if "is_ablated" not in sheet.edge_df:
        sheet.edge_df["is_ablated"] = 0
    ablated = shared_half_edges(sheet, faceA, faceB)
    if ablated.empty:
        raise ValueError(f"Faces {faceA} and {faceB} do not share an edge")
    sheet.edge_df.loc[ablated, "is_ablated"] = 1
    return ablated


def recoil_model(x, initial_recoil, K):
    """
    Kelvin-Voigt recoil of an ablated edge (pyVertexModel analyse_simulation.recoil_model):
        (L(t) - L0) / L0 = initial_recoil / K (1 - exp(-K t))
    :param x: time after the ablation (s)
    :param initial_recoil: initial strain rate (1/s)
    :param K: relaxation rate, elasticity / viscosity (1/s)
    """
    return (initial_recoil / K) * (1 - np.exp(-K * x))


def fit_ablation_equation(edge_length, time_steps):
    """
    Fit the Kelvin-Voigt recoil to an edge length time course, as pyVertexModel
    analyse_simulation.fit_ablation_equation (Veronika Lachina): the lengths are normalised by
    the FIRST RECORDED length (one time step after the ablation), not by the pre-ablation one
    :param edge_length: edge length at each time step after the ablation
    :param time_steps: time after the ablation (s)
    :return: K (1/s), initial_recoil (1/s), error bars of (initial_recoil, K)
    """
    edge_length = np.asarray(edge_length, dtype=float)
    strain = (edge_length - edge_length[0]) / edge_length[0]
    params, covariance = curve_fit(recoil_model, np.asarray(time_steps, dtype=float), strain,
                                   p0=[0.00001, 3], bounds=(0, np.inf))
    initial_recoil, K = params
    return K, initial_recoil, np.sqrt(np.diag(covariance))


def edges_to_ablate(sheet, n_ablations, seed=0):
    """
    Edges ablated to measure the recoil (pyVertexModel analyse_edge_recoil): the central cell
    with each of its neighbours, in a random order (seed), cycling if there are fewer neighbours
    than ablations
    :return: list of (cell, neighbour)
    """
    cell = central_faces(sheet, 1)[0]
    opp_face = opposite_faces(sheet)
    neighbours = np.unique(opp_face[(sheet.edge_df["face"].to_numpy() == cell) & (opp_face >= 0)])
    np.random.default_rng(seed).shuffle(neighbours)
    return [(int(cell), int(neighbours[i % len(neighbours)])) for i in range(n_ablations)]


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
