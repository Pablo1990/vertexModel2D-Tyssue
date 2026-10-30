import json
import os
import warnings

import numpy as np
import pandas as pd

# Import the project modules first: src/__init__.py silences the
# "install tyssue[viz]" warnings (ipyvolume/vispy are not needed here)
import src.auxFunctions as auxFunctions
import src.inputMechanicalParameters as inputMechanicalParameters
import src.vertexModel as vertexModel
import src.wound as wound

from tyssue.draw import sheet_view

# Ignore FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

HEIGHTS = [0.15, 1.5, 7.5, 15.0, 30.0]
# vertexModel.FARHADIFAR: volume + line tension + perimeter contractility
# vertexModel.EFFECTIVE_2D: apical = basal polygon with fixed height, V = hA, lateral area hP recoils
MODEL = vertexModel.EFFECTIVE_2D
FIXED_HEIGHT = True  # True: vertices only move in the plane (V = hA with a constant h). Always True for EFFECTIVE_2D
PERIODIC = True
NUM_CELLS = 200
SEED = 0  # random T1 detachments: same seed, same simulation
PURSE_STRING_TIME = 0.1  # minutes after ablation where the required purse string is computed

# Each height folder keeps what is needed to reproduce or continue its simulation:
#   simulation.json                 inputs (height, height mode, tissue, seed, calibrated params)
#   relaxed_state.hf5 / _specs.json tissue before the ablation
#   final_state.hf5 / _specs.json   tissue at the end of the wound healing
# Load a state with: [cellMap, geom, model] = vertexModel.load_state(os.path.join(folder, 'relaxed_state'))

# Calibration with experimental data, at the wing disc height
INPUT_FOLDER = 'input'
WOUND_AREA_FILE = os.path.join(INPUT_FOLDER, 'wound_area.csv')  # columns: time_after_ablation (min), wound_area
# Parameter calibrated for each model and its search range
CALIBRATION = {
    vertexModel.FARHADIFAR: ('line_tension_norm', (1e-3, 1.0)),  # or 'contractility_norm', 'lambdaV'
    vertexModel.EFFECTIVE_2D: ('lambdaS', (1e-2, 1e3)),  # lateral surface, or 'lambdaV'
}
CALIBRATED_PARAMETER, CALIBRATION_BOUNDS = CALIBRATION[MODEL]
RESULTS_FOLDER = os.path.join('results', 'wound_healing', MODEL)
TIME_AFTER_ABLATION = 6.0  # minutes, wound area compared at this time
EDGE_RECOIL = None  # initial recoil velocity after an edge ablation (length / min); None: not used

purse_strings = []
for height in HEIGHTS:
    print(f"---- Height {height} ----")
    output_folder = os.path.join(RESULTS_FOLDER, str(height))
    output_folder_relaxed = os.path.join(output_folder, 'relaxing')
    output_folder_ablation = os.path.join(output_folder, 'ablation')
    if not os.path.exists(output_folder_ablation):
        os.makedirs(output_folder_ablation)
        os.makedirs(output_folder_relaxed)

    # Calibrated values, saved once and used by every following run (delete the file to recalibrate)
    CALIBRATED_PARAMETERS_FILE = os.path.join(output_folder, 'calibrated_parameters.json')

    if os.path.exists(CALIBRATED_PARAMETERS_FILE):
        with open(CALIBRATED_PARAMETERS_FILE) as f:
            params = json.load(f)
        print(f"Using the calibrated parameters {params} from {CALIBRATED_PARAMETERS_FILE}")
    else:
        if EDGE_RECOIL is not None:
            targets = {'edge_recoil': EDGE_RECOIL}
        else:
            targets = {'wound_area': 165}

        [best_value, calibration] = vertexModel.calibrate(height, targets, CALIBRATED_PARAMETER,
                                                          bounds=CALIBRATION_BOUNDS,
                                                          fixedHeight=FIXED_HEIGHT,
                                                          timeAfterAblation=TIME_AFTER_ABLATION,
                                                          modelType=MODEL)
        params = {CALIBRATED_PARAMETER: float(best_value)}
        os.makedirs(os.path.dirname(CALIBRATED_PARAMETERS_FILE), exist_ok=True)
        calibration.to_csv(os.path.join(output_folder, 'calibration.csv'), index=False)
        with open(CALIBRATED_PARAMETERS_FILE, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Calibrated {params} for targets {targets}, saved in {CALIBRATED_PARAMETERS_FILE}")

    # Inputs needed to reproduce this simulation
    with open(os.path.join(output_folder, 'simulation.json'), 'w') as f:
        json.dump({'model': MODEL, 'height': height, 'fixed_height': FIXED_HEIGHT, 'periodic': PERIODIC,
                   'num_cells': NUM_CELLS, 'seed': SEED, 'params': params}, f, indent=2)
    np.random.seed(SEED)

    # Initialize Model
    [cellmap_init, geom, energyContributions_model] = vertexModel.initialize(numCellRows=NUM_CELLS, periodic=PERIODIC,
                                                                             fixedHeight=FIXED_HEIGHT, modelType=MODEL)

    # Update mechanical parameters
    cellmap_init = inputMechanicalParameters.update(cellmap_init, cell_height=height, params=params)

    # Initial stage
    energyContributions_model.compute_energy(cellmap_init)

    # Relax geometry
    [cellmap_relaxed, geom, model_relaxed, history_relaxed, relax_timecourse] = vertexModel.solve_until_relaxed(cellmap_init, geom, energyContributions_model)
    relax_timecourse.to_csv(os.path.join(output_folder, 'relax_timecourse.csv'), index=False)
    vertexModel.save_state(cellmap_relaxed, os.path.join(output_folder, 'relaxed_state'))

    # Purse string needed for the wound to start closing, PURSE_STRING_TIME minutes after ablation
    purse_string = vertexModel.required_purse_string(cellmap_relaxed, geom, model_relaxed, PURSE_STRING_TIME)
    purse_strings.append({'height': height, **purse_string})
    pd.DataFrame(purse_strings).to_csv(os.path.join(RESULTS_FOLDER, 'required_purse_string.csv'), index=False)
    print(height, purse_string)
    print(height, relax_timecourse.iloc[[0, -1]][['time', 'height', 'area', 'aspect_ratio']].to_string(index=False))

    fig, ax = sheet_view(cellmap_relaxed, ['y', 'x'], edge={"color": 1, 'colormap': 'Greys'})

    # color = np.array(cellmap_H.face_df.index.array, dtype=np.float32)
    # auxFunctions.create_frames(history_H, './results', edge={'color':'black'}, face={'visible': True , 'color':color, 'colormap': 'Greys'})
    auxFunctions.create_frames(history_relaxed, output_folder_relaxed, edge={'color': 'black'})

    # Wound healing, starting from the relaxed tissue
    [cellmap_H, geom, model_H, history_H, wound_timecourse] = vertexModel.solve_with_ablation(cellmap_relaxed, geom, model_relaxed)
    wound_timecourse.to_csv(os.path.join(output_folder, 'wound_timecourse.csv'), index=False)
    vertexModel.save_state(cellmap_H, os.path.join(output_folder, 'final_state'))
    print(height, wound.wound_features(wound_timecourse))

    fig, ax = sheet_view(cellmap_H, ['y', 'x'], edge={"color": 1, 'colormap': 'Greys'})

    #color = np.array(cellmap_H.face_df.index.array, dtype=np.float32)
    #auxFunctions.create_frames(history_H, './results', edge={'color':'black'}, face={'visible': True , 'color':color, 'colormap': 'Greys'})
    auxFunctions.create_frames(history_H, output_folder_ablation, edge={'color': 'black'})
    # auxFunctions.exportToMesh(history_H, './results')