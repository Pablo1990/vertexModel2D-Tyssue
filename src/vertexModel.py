from tyssue import config
import auxFunctions

def runModel(cellmap_original, geom_original, model_original, history_original, face_elasticity,
                 prefered_area, lower_line_tension, higher_line_tension, endTime):
    
   
    cellmap_original.update_specs(config.dynamics.quasistatic_plane_spec())
    #history_original = History(cellmap_original, extra_cols={"edge":["dx", "dy"]})

    cellmap_original.face_df["face_elasticity"] = face_elasticity
    cellmap_original.face_df["prefered_area"] = cellmap_original.face_df["area"].mean()*prefered_area

    cellmap_original.vert_df["viscosity"] = 1    
    
    cellmap_original.face_df["perimeter_elasticity"] = 0
    cellmap_original.face_df["contractility"] = 0
    
    cellmap_original = auxFunctions.line_tension_range(cellmap_original, lower_line_tension, higher_line_tension)

    
    
    solver1 = EulerSolver(cellmap_original, geom_original, model_original, history=history_original, auto_reconnect=True)
    res1 = solver1.solve(tf=endTime, dt=0.05)
    
    cellmap_new = copy.deepcopy(cellmap_original)
    geom_new = copy.deepcopy(geom_original)
    model_new = copy.deepcopy(model_original)
    history_new = copy.deepcopy(history_original)
    
    return [cellmap_new, geom_new, model_new, history_new]