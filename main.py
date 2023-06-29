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

##### Own functions
import src.vertexModel as vertexModel
import src.inputMechanicalParameters as inputMechanicalParameters


#### Initialize Model
[cellmap_init, geom_init, energyContributions_model] = vertexModel.initialize()

## Update mechanical parameters
cellmap_init = inputMechanicalParameters.update(cellmap_init)

## Initial stage
energyContributions_model.compute_energy(cellmap_init)

## RUN
[cellmap_H, geom_H, model_H, history_H] = vertexModel.solve(cellmap_init, geom_init, energyContributions_model, endTime = 30)