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


#### Initialize Model

## Definition of the sheet
# PlanarGeometry: Geometry methods for 2D planar cell arangements
# SheetGeometry: Geometry definitions for 2D sheets in 3D
# BulkGeometry: Geometry functions for 3D cell arangements
geom_init  = PlanarGeometry

## Defining energy contributions
# https://tyssue.readthedocs.io/en/latest/_modules/tyssue/dynamics/effectors.html
model_init = model_factory([    
    effectors.FaceAreaElasticity,
    effectors.LineTension,
    #effectors.LengthElasticity,
    effectors.PerimeterElasticity,
    #effectors.CellAreaElasticity,
    #effectors.FaceContractility,
    effectors.BarrierElasticity
    #effectors.LineViscosity
    ])

## Size of the patch
# noise = 0 -> hexagonal pattern
# noise = 1 -> random voronoi
cellmap_init = Sheet.planar_sheet_2d('tissue', 20, 20, 1, 1, noise=0.3)
cellmap_init.remove(cellmap_init.cut_out([[1, 20], [1, 20]]), trim_borders=True)

## Update geometry with the patch
geom_init.update_all(cellmap_init)
cellmap_init.reset_index()
cellmap_init.reset_topo()
fig, ax = sheet_view(cellmap_init, mode="quick")

## Init history object
history_init = History(cellmap_init, extra_cols={"edge":["dx", "dy"]})

## RUN
[cellmap_H, geom_H, model_H, history_H] = vertexModel.runModel(cellmap_init, geom_init, model_init, history_init, 
                                                       face_elasticity = 10, prefered_area = 0.6, lower_line_tension = 1, 
                                                       higher_line_tension = 100, endTime = 30)