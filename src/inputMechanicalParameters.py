import src.auxFunctions as auxFunctions

def update(cellmap):
    ## Per cell:

    ## Per face:
    cellmap.face_df["area_elasticity"] = 1
    cellmap.face_df["prefered_area"] = cellmap.face_df["area"].mean()

    cellmap.face_df["perimeter"] = 1
    cellmap.face_df["perimeter_elasticity"] = 0.1
    cellmap.face_df["prefered_perimeter"] = 3.81

    ## Per edge:
    rangeLineTension = 0

    if rangeLineTension:
        lower_line_tension = 1
        higher_line_tension = 10
        cellmap = line_tension_range(cellmap, lower_line_tension, higher_line_tension)
    else:
        cellmap.edge_df["line_tension"] = 1.0

    ## Per vertex:
    cellmap.vert_df["viscosity"] = 1 

    ## return
    return cellmap