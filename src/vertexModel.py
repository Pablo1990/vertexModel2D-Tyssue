import copy

import numpy as np
from tyssue import SheetGeometry, Sheet, History
from tyssue.behaviors import EventManager
from tyssue.draw import sheet_view
from tyssue.dynamics import effectors, model_factory
from tyssue.solvers import QSSolver
from tyssue.solvers.viscous import EulerSolver

import src.brownianMotion as brownianMotion
from effectors import EFFECTORS
from periodic import periodic_voronoi_sheet


def initialize(numCellRows=100, noiseCellShape=0.27, periodic=False):
    """
    Initialize the model
    :return:
    """
    ## Defining energy contributions
    # https://tyssue.readthedocs.io/en/latest/_modules/tyssue/dynamics/effectors.html
    energyContributions_model = model_factory([
        effectors.FaceVolumeElasticity,
        # effectors.LineTension,
        # effectors.LengthElasticity,
        effectors.PerimeterElasticity,
        # effectors.CellAreaElasticity,
        # effectors.FaceContractility,
        # effectors.BarrierElasticity
        # effectors.LineViscosity
        # effectors.BorderElasticity
        EFFECTORS[0],
    ])

    if not periodic:
        # noise = 0 -> hexagonal pattern
        # noise = 1 -> random voronoi
        cellMap = Sheet.planar_sheet_3d('tissue',
                                        nx=numCellRows,  # approximate number of cells on the x axis
                                        ny=numCellRows,  # approximate number of cells along the y axis
                                        distx=1,  # distance between 2 cells along x
                                        disty=1,  # distance between 2 cells along y
                                        noise=noiseCellShape)
    else:
        cellMap = periodic_voronoi_sheet(n_cells=a.n_cells, cell_area=1.0, seed=a.seed)

    try:
        cellMap.remove(cellMap.cut_out([[1, numCellRows], [1, numCellRows]]), trim_borders=True)
    except Exception:
        cellMap.sanitize(trim_borders=True, order_edges=True)
    cellMap.reset_index()
    cellMap.reset_topo()

    # Definition of the geometry of the sheet
    # PlanarGeometry: Geometry methods for 2D planar cell arangements
    # SheetGeometry: Geometry definitions for 2D sheets in 3D
    # BulkGeometry: Geometry functions for 3D cell arangements
    geom = SheetGeometry

    # Update geometry with the patch
    geom.update_all(cellMap)

    # Visualize the sheet
    fig, ax = sheet_view(cellMap, mode="quick")

    ## Connect cells with energy contributions
    cellMap.update_specs(energyContributions_model.specs)

    return [cellMap, geom, energyContributions_model]


def on_topo_change(sheet):
    """

    :param sheet:
    :return:
    """
    print('Topology changed!')


def solveEuler(cellMap, geom, energyContributions_model, endTime, dt=1, with_t3=False):
    """

    :param cellMap:
    :param geom:
    :param energyContributions_model:
    :param endTime:
    :param dt: time step
    :param with_t3: if True, perform type 3 transitions (a border vertex
        merging into the free edge of a neighbouring cell, new in vivatyssue).
        Useful for wound borders / tissue edges.
    :return:
    """
    ## Init history object
    # The History object records all the time steps
    history_cellMap = History(cellMap)

    ## Manager Initialization
    manager = EventManager("face", )
    # manager.append(basic_events.auto_reconnect)

    ## Init solver
    solver1 = EulerSolver(cellMap,
                          geom,
                          energyContributions_model,
                          manager=manager,
                          bounds=(
                              -cellMap.edge_df.length.median() / 10,
                              cellMap.edge_df.length.median() / 10
                          ),
                          history=history_cellMap,
                          auto_reconnect=True,
                          with_t3=with_t3)

    manager.update()

    ## Run the solver
    res1 = solver1.solve(tf=endTime, dt=dt, on_topo_change=on_topo_change,
                         topo_change_args=(solver1.eptm,))

    ## Deep copy to return it and being able to modify maintaining the previous one
    cellMap_new = copy.deepcopy(cellMap)
    history_new = copy.deepcopy(history_cellMap)

    return [cellMap_new, geom, energyContributions_model, history_new]


def solveStepByStep(cellMap, geom, energyContributions_model, endTime):
    """
    Quasi-static simulation: at each step, execute the scheduled events and
    then find the energy minimum (with T1 and T2 transitions).

    :param cellMap:
    :param geom:
    :param energyContributions_model:
    :param endTime:
    :return:
    """
    ## Init history object
    # The History object records all the time steps
    history_cellMap = History(cellMap)

    ## Manager Initialization
    manager = EventManager("face")

    ## Quasi-static solver
    # https://tyssue.readthedocs.io/en/latest/notebooks/07-EventManager.html
    solver = QSSolver(with_t1=True, with_t2=True)

    t = 0
    while t < endTime:
        # Execute the event in the current list
        manager.execute(cellMap)
        t += 1
        cellMap.reset_index(order=True)
        # Find energy min
        res = solver.find_energy_min(cellMap, geom, energyContributions_model)
        history_cellMap.record(time_stamp=t)
        # Switch event list from the next list to the current list
        manager.update()

    ## Deep copy to return it and being able to modify maintaining the previous one
    cellMap_new = copy.deepcopy(cellMap)
    history_new = copy.deepcopy(history_cellMap)

    return [cellMap_new, geom, energyContributions_model, history_new]


def compute_polygon_distribution(cell_map):
    """
    Compute the polygon distribution (number of neighbouring cells, from 4 to 9)
    of the cell map
    :param cell_map:
    :return: fraction of cells with 4, 5, ..., 9 neighbours
    """
    faces_neighbours = np.array(
        [len(cell_map.get_neighbors(face)) for face in cell_map.face_df.index]
    )
    sides = np.arange(4, 10)
    counts = np.array([(faces_neighbours == n).sum() for n in sides])
    polygon_distribution = counts / counts.sum()

    # Print the polygon distribution
    for n, frac in zip(sides, polygon_distribution):
        print(f"Polygons with {n} sides: {100 * frac:.1f}%")

    return polygon_distribution
