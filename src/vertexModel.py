from tyssue import config
from tyssue import PlanarGeometry, Sheet, History
from tyssue.draw import sheet_view
from tyssue.dynamics import effectors, model_factory
from tyssue.solvers.viscous import EulerSolver
import copy

import src.inputMechanicalParameters as inputMechanicalParameters

def initialize():
    ## Defining energy contributions
    # https://tyssue.readthedocs.io/en/latest/_modules/tyssue/dynamics/effectors.html
    energyContributions_model = model_factory([    
        effectors.FaceAreaElasticity,
        effectors.LineTension,
        #effectors.LengthElasticity,
        effectors.PerimeterElasticity,
        #effectors.CellAreaElasticity,
        #effectors.FaceContractility,
        #effectors.BarrierElasticity
        #effectors.LineViscosity
        ])

    ## Size of the patch
    numCellRows = 40
    noiseCellShape = 0.3

    # noise = 0 -> hexagonal pattern
    # noise = 1 -> random voronoi
    cellMap = Sheet.planar_sheet_2d('tissue', numCellRows, numCellRows, 1, 1, noise=noiseCellShape)
    cellMap.remove(cellMap.cut_out([[1, numCellRows], [1, numCellRows]]), trim_borders=True)
    cellMap.reset_index()
    cellMap.reset_topo()

    ## Definition of the geometry of the sheet
    # PlanarGeometry: Geometry methods for 2D planar cell arangements
    # SheetGeometry: Geometry definitions for 2D sheets in 3D
    # BulkGeometry: Geometry functions for 3D cell arangements
    geom  = PlanarGeometry

    # Update geometry with the patch
    geom.update_all(cellMap)

    # Visualize the sheet
    fig, ax = sheet_view(cellMap, mode="quick")

    ## Connect cells with energy contributions
    cellMap.update_specs(energyContributions_model.specs)

    return [cellMap, geom, energyContributions_model]


def solveEuler(cellMap, geom, energyContributions_model, endTime):

    ## Init history object
    # The History object records all the time steps
    history_cellMap = History(cellMap)

    ## Manager Initialization
    manager = EventManager("manager", )
    manager.append

    #type1_transition(cellMap, edge01, *, remove_tri_faces=True, multiplier=1.5)
    #resolve_t1s(cellMap, geom, model, solver, max_iter=60)

    ## Find the minima
    solver1 = EulerSolver(cellMap, 
        geom, 
        energyContributions_model,
        manager=manager,
        bounds=(
            -cellMap.edge_df.length.median()/10,
            cellMap.edge_df.length.median()/10
            ), 
        history=history_cellMap, 
        auto_reconnect=True)

    ## Deep copy to return it and being able to modify maintaining the previous one
    cellMap_new = copy.deepcopy(cellMap)
    history_new = copy.deepcopy(history_cellMap)

    return [cellMap_new, geom, energyContributions_model, history_new]

def solveStepByStep(cellMap, geom, energyContributions_model, endTime):

    ## Init history object
    # The History object records all the time steps
    history_cellMap = History(cellMap)

    ## Manager Initialization
    manager = EventManager("manager", )

    ## Find the minima in different timeSteps:
    # https://tyssue.readthedocs.io/en/latest/notebooks/07-EventManager.html
    t = 0
    stop = 30

    history_cellMap = history_cellMap(cellMap)

    while manager.current and t < stop:
        # Execute the event in the current list
        manager.execute(cellMap)
        t += 1
        cellMap.reset_index(order=True)
        # Find energy min
        res = solver.resolve_t1s(cellMap, geom, energyContributions_model)
        history_cellMap.record()
        # Switch event list from the next list to the current list
        manager.update()

    ## Deep copy to return it and being able to modify maintaining the previous one
    cellMap_new = copy.deepcopy(cellMap)
    history_new = copy.deepcopy(history_cellMap)

    return [cellMap_new, geom, energyContributions_model, history_new]