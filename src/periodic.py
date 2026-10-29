"""
Periodic planar tissue for the effective 2D model.

The geometry is tyssue's own: PlanarGeometry handles `settings['boundaries']`
(edge vectors, centroids and areas of cells crossing the box; see
vivatyssue/tests/geometry/test_periodic_geometry.py). This module adds:

- periodic_store_sheet: tyssue's stored periodic tissues (default planar_periodic8x8)
- periodic_voronoi_sheet: a Lloyd-relaxed Voronoi tissue on a torus (no free border).
- periodic_reconnect: EventManager behaviour doing tyssue's T1s (merge short edges +
  stochastic detachment). tyssue's collapse_edge / split_vert use raw vertex and face
  positions, so an edge or face crossing the box would put the new vertex on the wrong
  side; here positions are taken as minimum images first.
"""
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from tyssue import PlanarGeometry, Sheet, config
from tyssue.topology.base_topology import collapse_edge
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.utils import connectivity


def _periods(sheet):
    b = sheet.settings.get("boundaries")
    if not b:
        return {}
    return {u: (lim[0], lim[1] - lim[0]) for u, lim in b.items()}


def wrap(sheet):
    for u, (lo, L) in _periods(sheet).items():
        sheet.vert_df[u] = lo + np.mod(sheet.vert_df[u] - lo, L)


def periodic_voronoi_sheet(n_cells=100, cell_area=1.0, lloyd_steps=30, seed=0, identifier="periodic"):
    """Voronoi tissue of n_cells on a square torus of side sqrt(n_cells * cell_area)."""
    rng = np.random.default_rng(seed)
    L = np.sqrt(n_cells * cell_area)
    pts = rng.uniform(0, L, size=(n_cells, 2))
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]) * L

    for _ in range(lloyd_steps + 1):
        tiled = np.concatenate([pts + s for s in shifts])
        vor = Voronoi(tiled)
        centre_block = 4 * n_cells  # shifts[4] == (0, 0)
        regions = [vor.regions[vor.point_region[centre_block + i]] for i in range(n_cells)]
        polys = [vor.vertices[r] for r in regions]
        if _ < lloyd_steps:
            pts = np.mod(np.array([_poly_centroid(p) for p in polys]), L)

    # unique vertices on the torus
    keys, vert_xy, srce, trgt, face = {}, [], [], [], []
    for f, poly in enumerate(polys):
        if _signed_area(poly) < 0:
            poly = poly[::-1]
        ids = []
        for v in poly:
            w = np.mod(v, L)
            k = tuple(np.round(w / L * 1e7).astype(np.int64) % int(1e7))
            if k not in keys:
                keys[k] = len(vert_xy)
                vert_xy.append(w)
            ids.append(keys[k])
        for a, b in zip(ids, ids[1:] + ids[:1]):
            srce.append(a)
            trgt.append(b)
            face.append(f)

    specs = config.geometry.planar_periodic_sheet()
    specs["settings"]["boundaries"] = {"x": [0.0, L], "y": [0.0, L]}
    datasets = {
        "vert": pd.DataFrame(np.array(vert_xy), columns=["x", "y"]),
        "edge": pd.DataFrame({"srce": srce, "trgt": trgt, "face": face}),
        "face": pd.DataFrame(np.mod(pts, L), columns=["x", "y"]),
    }
    sheet = Sheet(identifier, datasets, specs)
    sheet.reset_index(order=True)
    sheet.reset_topo()
    PlanarGeometry.update_all(sheet)
    return sheet


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _poly_centroid(p):
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = 0.5 * cross.sum()
    return np.array([((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]) / (6 * a)


# ----------------------------------------------------------------------- T1s
def periodic_merge_vertices(sheet, geom=PlanarGeometry):
    """tyssue merge_vertices, but the target is moved next to the source (minimum image)
    before collapsing, so the merged vertex is not placed across the box."""
    d_min = sheet.settings.get("threshold_length", 1e-3)
    merged = False
    while True:
        short = sheet.edge_df.index[sheet.edge_df["length"].to_numpy() < d_min]
        if not len(short):
            break
        e = np.random.choice(short)
        s, t = sheet.edge_df.loc[e, ["srce", "trgt"]].astype(int)
        for u in sheet.coords:
            sheet.vert_df.loc[t, u] = sheet.vert_df.loc[s, u] + sheet.edge_df.loc[e, "d" + u]
        collapse_edge(sheet, e, allow_two_sided=False)
        geom.update_all(sheet)
        merged = True
    if merged:
        sheet.network_changed = True
    return merged


def periodic_detach_vertices(sheet, geom=PlanarGeometry):
    """tyssue detach_vertices (rank 4 with p_4, rank >= 5 with p_5p per unit time), with the
    face centre put next to the vertex before splitting."""
    rank = connectivity.vertex_rank(sheet)
    if rank.max() <= 3:
        return False
    dt = sheet.settings.get("dt", 1.0)
    p_4 = sheet.settings.get("p_4", 0.1) * dt
    p_5p = sheet.settings.get("p_5p", 1e-2) * dt
    rank4 = sheet.vert_df.index[rank == 4]
    rank5p = sheet.vert_df.index[rank > 4]
    to_detach = np.concatenate([rank4[np.random.random(rank4.size) < p_4],
                                rank5p[np.random.random(rank5p.size) < p_5p]])
    per = _periods(sheet)
    for vert in to_detach:
        faces = sheet.edge_df.loc[sheet.edge_df["srce"] == vert, "face"].to_numpy()
        if not len(faces):
            continue
        face = np.random.choice(faces)
        for u, (lo, L) in per.items():
            dv = sheet.face_df.loc[face, u] - sheet.vert_df.loc[vert, u]
            sheet.face_df.loc[face, u] = sheet.vert_df.loc[vert, u] + dv - L * np.round(dv / L)
        sheet_split(sheet, vert, face=face)
    if len(to_detach):
        wrap(sheet)
        geom.update_all(sheet)
        sheet.network_changed = True
    return bool(len(to_detach))


def periodic_reconnect(sheet, manager, **kwargs):
    """Drop-in for tyssue.behaviors.sheet.basic_events.reconnect (periodic-safe)."""
    sheet.settings.update(kwargs)
    periodic_merge_vertices(sheet)
    periodic_detach_vertices(sheet)
    manager.append(periodic_reconnect, **kwargs)
