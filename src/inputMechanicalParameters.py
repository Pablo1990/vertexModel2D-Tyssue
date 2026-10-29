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

    # Face Volume Elasticity
    # Energy 1/2 lambdaV ((V - V0) / V0)^2, as in the 3D model: vol_elasticity = lambdaV / V0^2
    # (set below, after the optional params, so lambdaV and prefered_vol can be calibrated)
    cellmap.settings["lambdaV"] = 1.0
    cellmap.face_df["prefered_vol"] = cellmap.face_df["area"].mean() * 0.95 * cell_height
    # SheetGeometry ('flat'): height = z - basal_shift, recomputed at every update
    cellmap.vert_df["basal_shift"] = cellmap.vert_df["z"] - cell_height

    # PerimeterElasticity
    cellmap.face_df["perimeter"] = 1
    cellmap.face_df["perimeter_elasticity"] = 1
    cellmap.face_df["prefered_perimeter"] = 3.81

    # Per vertex:
    cellmap.vert_df["viscosity"] = 0.07

    if params:
        cellmap = set_parameters(cellmap, params)

    cellmap.face_df["vol_elasticity"] = cellmap.settings["lambdaV"] / cellmap.face_df["prefered_vol"] ** 2

    return cellmap


def set_parameters(cellmap, params):
    """
    Overwrite parameters by name, e.g. when calibrating: a column of face_df, vert_df or
    edge_df if it exists, otherwise a setting
    :param cellmap:
    :param params: dict {name: value}, e.g. {'purseStringStrength': 2e-5, 'lambdaV': 0.5}
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

