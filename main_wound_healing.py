import os
import warnings

# Import the project modules first: src/__init__.py silences the
# "install tyssue[viz]" warnings (ipyvolume/vispy are not needed here)
import src.auxFunctions as auxFunctions
import src.inputMechanicalParameters as inputMechanicalParameters
import src.vertexModel as vertexModel

from tyssue.draw import sheet_view

# Ignore FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

HEIGHTS = [0.15, 1.5, 7.5, 15.0, 30.0]

for height in HEIGHTS:
    output_folder = os.path.join('results', 'wound_healing', str(height))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize Model
    [cellmap_init, geom, energyContributions_model] = vertexModel.initialize(periodic=True)

    # Update mechanical parameters
    cellmap_init = inputMechanicalParameters.update(cellmap_init)

    # Initial stage
    energyContributions_model.compute_energy(cellmap_init)

    # RUN
    # with_t3=True enables type 3 transitions at free borders (e.g. wound edges)
    [cellmap_H, geom, model_H, history_H] = vertexModel.solveEuler(cellmap_init, geom, energyContributions_model,
                                                                   endTime=cellmap_init.settings["tend"], with_t3=False)

    fig, ax = sheet_view(cellmap_H, ['y', 'x'], edge={"color": 1, 'colormap': 'Greys'})

    #color = np.array(cellmap_H.face_df.index.array, dtype=np.float32)
    #auxFunctions.create_frames(history_H, './results', edge={'color':'black'}, face={'visible': True , 'color':color, 'colormap': 'Greys'})
    auxFunctions.create_frames(history_H, output_folder, edge={'color': 'black'})
    # auxFunctions.exportToMesh(history_H, './results')