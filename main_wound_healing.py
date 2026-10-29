import os
import warnings

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
FIXED_HEIGHT = True  # True: vertices only move in the plane (V = hA with a constant h)

for height in HEIGHTS:
    output_folder = os.path.join('results', 'wound_healing', str(height))
    output_folder_relaxed = os.path.join(output_folder, 'relaxing')
    output_folder_ablation = os.path.join(output_folder, 'ablation')
    if not os.path.exists(output_folder_ablation):
        os.makedirs(output_folder_ablation)
        os.makedirs(output_folder_relaxed)

    # Initialize Model
    [cellmap_init, geom, energyContributions_model] = vertexModel.initialize(periodic=True, fixedHeight=FIXED_HEIGHT)

    # Update mechanical parameters
    cellmap_init = inputMechanicalParameters.update(cellmap_init, cell_height=height)

    # Initial stage
    energyContributions_model.compute_energy(cellmap_init)

    # Relax geometry
    [cellmap_relaxed, geom, model_relaxed, history_relaxed, relax_timecourse] = vertexModel.solve_until_relaxed(cellmap_init, geom, energyContributions_model)
    relax_timecourse.to_csv(os.path.join(output_folder, 'relax_timecourse.csv'), index=False)
    print(height, relax_timecourse.iloc[[0, -1]][['time', 'height', 'area', 'aspect_ratio']].to_string(index=False))

    fig, ax = sheet_view(cellmap_relaxed, ['y', 'x'], edge={"color": 1, 'colormap': 'Greys'})

    # color = np.array(cellmap_H.face_df.index.array, dtype=np.float32)
    # auxFunctions.create_frames(history_H, './results', edge={'color':'black'}, face={'visible': True , 'color':color, 'colormap': 'Greys'})
    auxFunctions.create_frames(history_relaxed, output_folder_relaxed, edge={'color': 'black'})

    # Wound healing, starting from the relaxed tissue
    [cellmap_H, geom, model_H, history_H, wound_timecourse] = vertexModel.solve_with_ablation(cellmap_relaxed, geom, model_relaxed)
    wound_timecourse.to_csv(os.path.join(output_folder, 'wound_timecourse.csv'), index=False)
    print(height, wound.wound_features(wound_timecourse))

    fig, ax = sheet_view(cellmap_H, ['y', 'x'], edge={"color": 1, 'colormap': 'Greys'})

    #color = np.array(cellmap_H.face_df.index.array, dtype=np.float32)
    #auxFunctions.create_frames(history_H, './results', edge={'color':'black'}, face={'visible': True , 'color':color, 'colormap': 'Greys'})
    auxFunctions.create_frames(history_H, output_folder_ablation, edge={'color': 'black'})
    # auxFunctions.exportToMesh(history_H, './results')