def update(cellmap):
    ## Per cell:

    ## Per face:
    cellmap.face_df["area_elasticity"] = 1
    cellmap.face_df["prefered_area"] = cellmap.face_df["area"].mean()

    cellmap.face_df["perimeter"] = 1
    cellmap.face_df["perimeter_elasticity"] = 0.1
    cellmap.face_df["prefered_perimeter"] = 3.81

    ## Per edge:
    cellmap.edge_df["line_tension"] = 1.0

    ## return
    return cellmap_init