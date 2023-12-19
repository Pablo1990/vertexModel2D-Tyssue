import warnings

from tyssue.draw import sheet_view

import src.auxFunctions as auxFunctions
import src.inputMechanicalParameters as inputMechanicalParameters
import src.vertexModel as vertexModel

# Ignore FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Model
[cellmap_init, geom, energyContributions_model] = vertexModel.initialize()

# Update mechanical parameters
cellmap_init = inputMechanicalParameters.update(cellmap_init)

# Initial stage
energyContributions_model.compute_energy(cellmap_init)

# RUN
[cellmap_H, geom, model_H, history_H] = vertexModel.solveEuler(cellmap_init, geom, energyContributions_model,
                                                               endTime=100)

fig, ax = sheet_view(cellmap_H, ['y', 'x'], edge={"color":1, 'colormap': 'Greys'})
auxFunctions.create_frames(history_H, './results', edge={'color': 'black'})
# auxFunctions.exportToMesh(history_H, './results')
