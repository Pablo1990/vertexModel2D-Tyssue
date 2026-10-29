import numpy as np

import src.auxFunctions as auxFunctions

def update(cellmap, cell_height=15.0):
    """

    :param cellmap:
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
    cellmap.face_df["vol_elasticity"] = 1.0
    cellmap.face_df["prefered_vol"] = cellmap.face_df["area"].mean() * 0.95
    cellmap.vert_df["height"] = cell_height

    # PerimeterElasticity
    cellmap.face_df["perimeter"] = 1
    cellmap.face_df["perimeter_elasticity"] = 1
    cellmap.face_df["prefered_perimeter"] = 3.81

    # Per vertex:
    cellmap.vert_df["viscosity"] = 0.07

    return cellmap

# --- Time profile of the wound contractility (KgContractility) -------------------
_PS_PROFILE = np.array([1.0, 0.96, 1.007, 1.74, 2.37, 2.61, 2.487, 2.536, 2.46, 2.52, 2.606, 2.606, 2.606,
                        2.606, 2.606, 2.606, 2.606, 2.606, 2.606, 2.606, 2.606]) - 1

def wound_contractility(cellmap, t):
    """(C_ps, C_lc) at time t, as get_contractility_based_on_location() in 3D."""
    t_after = t - cellmap.settings["TInitAblation"]
    if t_after < 0:
        return cellmap.settings["cLineTension"], cellmap.settings["p.cLineTension"]
    if cellmap.settings["TypeOfPurseString"] == 0:
        times = np.arange(0, 60 + 3, 3) / 60 * (cellmap.settings["TEndAblation"] - cellmap.settings["TInitAblation"])
        ps = np.interp(t_after, times, _PS_PROFILE / _PS_PROFILE.max() * cellmap.settings["purseStringStrength"])
        return ps
    if cellmap.settings["TypeOfPurseString"] == 3:  # constant
        return cellmap.settings["purseStringStrength"]

    raise NotImplementedError(f"TypeOfPurseString not found")

