import copy
import ipyvolume as ipv
import json
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import random
import sys
from IPython.display import Image
from scipy import optimize

##### From tyssue
from tyssue import PlanarGeometry, Sheet, SheetGeometry, History, EventManager
from tyssue.draw import sheet_view
from tyssue.draw.plt_draw import create_gif, plot_forces
from tyssue.dynamics import effectors, model_factory, PlanarModel
from tyssue.generation import three_faces_sheet
from tyssue.io import hdf5
from tyssue.solvers.quasistatic import QSSolver
from tyssue.solvers.viscous import EulerSolver
from tyssue.stores import load_datasets
from tyssue.topology.sheet_topology import remove_face, cell_division

##### Own functions
import vertexModel
import inputMechanicalParameters


#### Initialize Model
[cellmap_init] = vertexModel.initialize()

## Update mechanical parameters
cellmap_init = inputMechanicalParameters.update(cellmap_init)

## Initial stage
model_init.compute_energy(cellmap_init)

## RUN
[cellmap_H, geom_H, model_H, history_H] = vertexModel.solve(cellmap_init, geom_init, model_init, history_init, 
                                                       face_elasticity = 10, prefered_area = 0.6, lower_line_tension = 1, 
                                                       higher_line_tension = 100, endTime = 30)