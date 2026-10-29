import numpy as np

import src.auxFunctions as auxFunctions

def update(cellmap, cell_height=15.0, params=None):
    """
    Set the default mechanical parameters
    :param cellmap:
    :param cell_height:
    :param params: optional dict {name: value} overwriting the defaults (see set_parameters)
    :return:
    """
    # Stochastically detaches vertices from rosettes.
    # Uses two probabilities `p_4` and `p_5p`
    cellmap.settings["p_4"] = 10
    cellmap.settings["p_5p"] = 0.1
    cellmap.settings["threshold_length"] = 2e-2

    # Additional settings
    cellmap.settings["n_ablated"] = 10 # cells ablated around the tissue centre
    cellmap.settings["TInitAblation"] = 20.0 # minutes
    cellmap.settings["TEndAblation"] = 80.0 # minutes
    cellmap.settings["tend"] = 80.0 # minutes
    cellmap.settings["dt"] = 0.01 # minutes
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
    # SheetGeometry ('flat'): height = z - basal_shift, recomputed at every update
    cellmap.vert_df["basal_shift"] = cellmap.vert_df["z"] - cell_height

    # Per vertex:
    cellmap.vert_df["viscosity"] = 0.07

    if params:
        cellmap = set_parameters(cellmap, params)

    set_farhadifar_parameters(cellmap)

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

