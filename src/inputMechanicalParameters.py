import src.auxFunctions as auxFunctions


def update(cellmap):
    """

    :param cellmap:
    :return:
    """
    # Per sheet:
    cellmap.settings["temperature"] = 10
    # Stochastically detaches vertices from rosettes.
    # Uses two probabilities `p_4` and `p_5p`
    cellmap.settings["p_4"] = 10
    cellmap.settings["p_5p"] = 0.1
    cellmap.settings["threshold_length"] = 2e-2

    # Per cell:

    # Per face:
    cellmap.face_df["area_elasticity"] = 50
    cellmap.face_df["prefered_area"] = cellmap.face_df["area"].mean() * 1.1

    cellmap.face_df["perimeter"] = 1
    cellmap.face_df["perimeter_elasticity"] = 10
    cellmap.face_df["prefered_perimeter"] = 3.81

    # Per edge:
    rangeLineTension = 0

    if rangeLineTension:
        lower_line_tension = 0.1
        higher_line_tension = 1
        cellmap = auxFunctions.line_tension_range(cellmap, lower_line_tension, higher_line_tension)
    else:
        cellmap.edge_df["line_tension"] = 0.1

    # Per vertex:
    cellmap.vert_df["viscosity"] = 1

    return cellmap
