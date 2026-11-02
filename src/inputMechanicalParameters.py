import numpy as np
from tyssue import SheetGeometry

import src.auxFunctions as auxFunctions

DEFAULT_DT = 0.01  # minutes
# Mean cell volume of every pyVertexModel tissue (Geo.resize_tissue default): whatever the height,
# the 3D tissue is scaled isotropically to this volume
REFERENCE_CELL_VOLUME = 1 #0.0003168604676977124

def update(cellmap, cell_height=15.0, params=None):
    """
    Resize the tissue to the pyVertexModel reference volume (resize_tissue) and set the default
    mechanical parameters
    :param cellmap: tissue from vertexModel.initialize (not resized yet)
    :param cell_height: cell height before resizing, i.e. the aspect ratio (height / sqrt(area)) of
        the cells times the square root of the initial mean area (~1). The actual height is
        cell_height * settings['length_scale']
    :param params: optional dict {name: value} overwriting the defaults (see set_parameters)
    :return:
    """
    cell_height = resize_tissue(cellmap, cell_height)

    # Stochastically detaches vertices from rosettes.
    # Uses two probabilities `p_4` and `p_5p`
    cellmap.settings["p_4"] = 10
    cellmap.settings["p_5p"] = 0.1
    cellmap.settings["threshold_length"] = 2e-2 * cellmap.settings["length_scale"]  # as the cell size

    # Additional settings
    cellmap.settings["n_ablated"] = 10 # cells ablated around the tissue centre
    cellmap.settings["TInitAblation"] = 20.0 # minutes
    cellmap.settings["TEndAblation"] = 80.0 # minutes
    cellmap.settings["tend"] = 80.0 # minutes
    cellmap.settings["dt"] = DEFAULT_DT
    cellmap.settings["record_every"] = 1 # minutes

    # Wound contractility (TypeOfPurseString 0 = intensity-based profile)
    cellmap.settings["TypeOfPurseString"] = 0
    cellmap.settings["purseStringStrength"] = 1e-5
    cellmap.settings["cLineTension"] = 0.0

    # Farhadifar et al. 2007, with the volume V = hA in place of the area:
    #   E = sum lambdaV/2 ((V - V0)/V0)^2 + sum Lambda l + sum Gamma/2 P^2
    # For a fixed height, the volume term is K/2 (A - A0)^2 with K = lambdaV / A0^2. The line
    # tension and contractility are given dimensionless as in Farhadifar:
    #   line_tension_norm = Lambda / (K A0^(3/2)),  contractility_norm = Gamma / (K A0)
    # (0.12 and 0.04: their case I). The actual values are set below, after the optional
    # params, so any of lambdaV, line_tension_norm or contractility_norm can be calibrated
    cellmap.settings["lambdaV"] = 1.0
    cellmap.settings["line_tension_norm"] = 0.12
    cellmap.settings["contractility_norm"] = 0
    cellmap.settings["prefered_area"] = cellmap.face_df["area"].mean()  # A0
    cellmap.face_df["prefered_vol"] = cellmap.settings["prefered_area"] * cell_height

    # Effective 2D model (apical = basal polygon, fixed height): lateral area hP, as the 3D
    # cell-based surface energy, E = lambdaS/2 ((hP - hP0)/S0)^2 with S0 = 2 A0 + h P0 and
    # P0 = lateral_ref * the mean perimeter (as ref_A0 in the 3D model)
    cellmap.settings["cell_height"] = cell_height
    cellmap.settings["lambdaS"] = 1.0
    cellmap.settings["lateral_ref"] = 0.95
    # SheetGeometry ('flat'): height = z - basal_shift, recomputed at every update
    cellmap.vert_df["basal_shift"] = cellmap.vert_df["z"] - cell_height

    # Per vertex:
    cellmap.vert_df["viscosity"] = 0.07

    if params:
        cellmap = set_parameters(cellmap, params)

    if cellmap.settings.get("model") in ("effective_2d", "effective_2d_tilted"):
        set_effective_2d_parameters(cellmap)
    else:
        set_farhadifar_parameters(cellmap)

    return cellmap


def resize_tissue(cellmap, cell_height, average_volume=REFERENCE_CELL_VOLUME):
    """
    Scale the tissue isotropically (x, y and the cell height) so that the mean volume of the alive
    cells, height * area, is average_volume, as pyVertexModel Geo.resize_tissue. The aspect ratio
    (height / sqrt(area)) is kept, and so is the shape of every cell. The periodic boundaries are
    scaled with the tissue, around the origin
    :param cellmap: tissue with z = 0, not resized yet
    :param cell_height: height before resizing
    :param average_volume: target mean cell volume
    :return: the cell height after resizing
    """
    if "length_scale" in cellmap.settings:
        raise ValueError("The tissue has already been resized (settings['length_scale'])")

    SheetGeometry.update_all(cellmap)
    meanArea = cellmap.face_df.loc[cellmap.face_df["is_alive"] == 1, "area"].mean()
    scale = (average_volume / (cell_height * meanArea)) ** (1 / 3)

    planarCoords = [coord for coord in cellmap.coords if coord != "z"]
    cellmap.vert_df[planarCoords] *= scale
    boundaries = cellmap.settings.get("boundaries")
    if boundaries:
        cellmap.settings["boundaries"] = {coord: [float(low * scale), float(high * scale)] for coord, (low, high) in boundaries.items()}
    cellmap.settings["length_scale"] = float(scale)
    cellmap.settings["aspect_ratio"] = float(cell_height / np.sqrt(meanArea))

    height = cell_height * scale
    cellmap.vert_df["basal_shift"] = cellmap.vert_df["z"] - height
    SheetGeometry.update_all(cellmap)
    return height


def set_effective_2d_parameters(cellmap):
    """
    Absolute values of the volume and lateral area parameters of the effective 2D model:
    lambdaS/2 ((A_lat - A_lat0)/S0)^2 with A_lat = h P, A_lat0 = h P0 and S0 = 2 A0 + h P0
    (LateralAreaElasticity with lateral_elasticity = lambdaS / S0^2)
    :param cellmap:
    :return:
    """
    h = cellmap.settings["cell_height"]
    area0 = cellmap.settings["prefered_area"]
    lateralArea0 = h * cellmap.settings["lateral_ref"] * cellmap.face_df["perimeter"].mean()
    surface0 = 2 * area0 + lateralArea0

    cellmap.face_df["vol_elasticity"] = cellmap.settings["lambdaV"] / cellmap.face_df["prefered_vol"] ** 2
    cellmap.face_df["prefered_lateral_area"] = lateralArea0
    cellmap.face_df["lateral_elasticity"] = cellmap.settings["lambdaS"] / surface0 ** 2
    return cellmap


def set_farhadifar_parameters(cellmap):
    """
    Absolute values of the volume, line tension and contractility parameters from lambdaV and
    the dimensionless line_tension_norm and contractility_norm (Farhadifar et al. 2007)
    :param cellmap:
    :return:
    """
    lambdaV = cellmap.settings["lambdaV"]
    area0 = cellmap.settings["prefered_area"]
    K = lambdaV / area0 ** 2

    cellmap.face_df["vol_elasticity"] = lambdaV / cellmap.face_df["prefered_vol"] ** 2
    cellmap.edge_df["line_tension"] = cellmap.settings["line_tension_norm"] * K * area0 ** 1.5
    cellmap.face_df["contractility"] = cellmap.settings["contractility_norm"] * K * area0
    return cellmap


def set_parameters(cellmap, params):
    """
    Overwrite parameters by name, e.g. when calibrating: a column of face_df, vert_df or
    edge_df if it exists, otherwise a setting
    :param cellmap:
    :param params: dict {name: value}, e.g. {'line_tension_norm': 0.2, 'lambdaV': 0.5}
    :return:
    """
    for name, value in params.items():
        for df in (cellmap.face_df, cellmap.vert_df, cellmap.edge_df):
            if name in df.columns:
                df[name] = value
                break
        else:
            cellmap.settings[name] = value
    return cellmap

# --- Time profile of the wound contractility (KgContractility) -------------------
_PS_PROFILE = np.array([1.0, 0.96, 1.007, 1.74, 2.37, 2.61, 2.487, 2.536, 2.46, 2.52, 2.606, 2.606, 2.606,
                        2.606, 2.606, 2.606, 2.606, 2.606, 2.606, 2.606, 2.606]) - 1

def wound_contractility(cellmap, t):
    """Purse string tension at time t, as get_contractility_based_on_location() in 3D."""
    t_after = t - cellmap.settings["TInitAblation"]
    if t_after < 0:
        return cellmap.settings["cLineTension"]
    if cellmap.settings["TypeOfPurseString"] == 0:
        times = np.arange(0, 60 + 3, 3) / 60 * (cellmap.settings["TEndAblation"] - cellmap.settings["TInitAblation"])
        ps = np.interp(t_after, times, _PS_PROFILE / _PS_PROFILE.max() * cellmap.settings["purseStringStrength"])
        return ps
    if cellmap.settings["TypeOfPurseString"] == 3:  # constant
        return cellmap.settings["purseStringStrength"]

    raise NotImplementedError(f"TypeOfPurseString not found")

