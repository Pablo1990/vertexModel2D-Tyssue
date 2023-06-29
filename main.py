import sys
import random
import numpy as np
import pandas as pd
import numpy as np
import json
import matplotlib.pylab as plt
import ipyvolume as ipv
import copy
from IPython.display import Image
from scipy import optimize

##### From tyssue
from tyssue import Sheet, SheetGeometry, History, EventManager
from tyssue import SheetGeometry as geom
from tyssue import PlanarGeometry as geom
from tyssue.draw.plt_draw import create_gif
from tyssue.draw.plt_draw import plot_forces
from tyssue.draw import sheet_view
from tyssue.dynamics import effectors, model_factory
from tyssue.dynamics import PlanarModel
from tyssue.dynamics import SheetModel as model
from tyssue.generation import three_faces_sheet
from tyssue.solvers.viscous import EulerSolver
from tyssue.solvers.quasistatic import QSSolver
from tyssue.draw import sheet_view
from tyssue.draw.plt_draw import plot_forces
from tyssue.io import hdf5
from tyssue.draw import sheet_view
from tyssue.geometry.planar_geometry import PlanarGeometry as geom
from tyssue.solvers.quasistatic import QSSolver
from tyssue.dynamics.planar_vertex_model import PlanarModel as model
from tyssue.stores import load_datasets
from tyssue.topology.sheet_topology import remove_face, cell_division

##### Own functions
import vertexModel


#### Definition of the sheet
geom_original  = SheetGeometry # This may need to change
model_original = model_factory([    
    effectors.FaceAreaElasticity,
    effectors.LineTension
    ]) 
cellmap_original = Sheet.planar_sheet_3d('cellmap_original', 20, 20, 1, 1) # This may need to change
cellmap_original.sanitize ( trim_borders=True, order_edges=True )
geom_original.update_all(cellmap_original)

history_original = History(cellmap_original, extra_cols={"edge":["dx", "dy"]})

## RUN
[cellmap_H, geom_H, model_H, history_H] = vertexModel.runModel(cellmap_original, geom_original, model_original, history_original, 
                                                       face_elasticity = 10, prefered_area = 0.6, lower_line_tension = 1, 
                                                       higher_line_tension = 100, endTime = 30)