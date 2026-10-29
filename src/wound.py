import numpy as np

from inputMechanicalParameters import wound_contractility
from tyssue import PlanarGeometry

geom = PlanarGeometry

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


def ablate(sheet, faces):
    """Combine the ablated cells into a single debris face (Geo.ablate_cells, combine_cells=True).

    The debris has no surface energy and no contractility (is_alive = 0); its
    3D volume stiffness is lambdaV * 1e-8, i.e. negligible, so it is omitted.
    """
    faces = np.asarray(faces)
    debris = faces[0]
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


def update_wound_tension(sheet, t=None, ps=None, lc=None):
    if ps is None:
        ps = wound_contractility(sheet, t)
    sheet.settings["wound_tension"] = ps


def wound_measures(sheet):
    d = debris_face(sheet)
    if d is None:
        return {"wound_area": 0.0, "wound_perimeter": 0.0, "num_cells_wound_edge": 0}
    we = wound_edges(sheet)
    return {
        "wound_area": float(abs(sheet.face_df.loc[d, "area"])),
        "wound_perimeter": float(sheet.face_df.loc[d, "perimeter"]),
        "num_cells_wound_edge": int(sheet.edge_df.loc[we, "face"].nunique()),
    }