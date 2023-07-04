import src.auxFunctions as auxFunctions

def update(cellmap):
    ## Per sheet:
    cellmap.settings["temperature"] = 2e-1
    cellmap.settings["p_4"] = 10.0
    cellmap.settings["p_5p"] = 1.0
    cellmap.settings["threshold_length"] = 2e-2

    ## Per cell:

    ## Per face:
    cellmap.face_df["area_elasticity"] = 5
    cellmap.face_df["prefered_area"] = cellmap.face_df["area"].mean()*1.2

    cellmap.face_df["perimeter"] = 1
    cellmap.face_df["perimeter_elasticity"] = 0.01
    cellmap.face_df["prefered_perimeter"] = 3.81

    ## Per edge:
    rangeLineTension = 0

    if rangeLineTension:
        lower_line_tension = 0.1
        higher_line_tension = 1
        cellmap = auxFunctions.line_tension_range(cellmap, lower_line_tension, higher_line_tension)
    else:
        cellmap.edge_df["line_tension"] = 0.5

    ## Per vertex:
    cellmap.vert_df["viscosity"] = 1 

    ## return
    return cellmap