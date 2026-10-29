import copy

import numpy as np
import pandas as pd
from tyssue import SheetGeometry, Sheet, History, config
from tyssue.behaviors import EventManager
from tyssue.draw import sheet_view
from tyssue.dynamics import effectors, model_factory
from tyssue.solvers import QSSolver
from tyssue.solvers.viscous import EulerSolver

import src.brownianMotion as brownianMotion
import src.wound as wound
from src.effectors import EFFECTORS
from src.periodic import periodic_reconnect, periodic_voronoi_sheet


def initialize(numCellRows=100, noiseCellShape=0.27, periodic=False, fixedHeight=False):
    """
    Initialize the model: a 2.5D apical sheet (SheetGeometry, 'flat'), where each vertex has a
    height = z - basal_shift and each cell a volume = height * area

    :param numCellRows:
    :param noiseCellShape:
    :param periodic: periodic tissue (no free border)
    :param fixedHeight: if True, the vertices only move in the plane, so the cell height is
        fixed (V = hA with a constant h). If False, the volume term also moves the vertices in z,
        so the cells change their height
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
        try:
            cellMap.remove(cellMap.cut_out([[1, numCellRows], [1, numCellRows]]), trim_borders=True)
        except Exception:
            cellMap.sanitize(trim_borders=True, order_edges=True)
    else:
        # periodic: no border to trim. The sheet is planar (x, y): add z, height and basal_shift
        cellMap = periodic_voronoi_sheet(n_cells=numCellRows, cell_area=1.0)
        cellMap.update_specs(config.geometry.sheet_spec())
        cellMap = Sheet(cellMap.identifier, cellMap.datasets, cellMap.specs)

    # Height measured along z from basal_shift (the default 'cylindrical' uses the distance to the z axis)
    cellMap.settings["geometry"] = "flat"
    cellMap.settings["fixed_height"] = fixedHeight
    cellMap.vert_df["z"] = 0.0
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

    if fixedHeight:
        energyContributions_model = freeze_height(energyContributions_model)

    return [cellMap, geom, energyContributions_model]


def freeze_height(energyContributions_model):
    """
    Keep only the in-plane forces, so the vertices don't move in z and the cell height is fixed

    :param energyContributions_model: model from model_factory
    :return:
    """
    compute_gradient = energyContributions_model.compute_gradient

    def in_plane_gradient(eptm, components=False):
        grad = compute_gradient(eptm, components)
        if not components:
            grad["gz"] = 0.0
        return grad

    energyContributions_model.compute_gradient = staticmethod(in_plane_gradient)
    return energyContributions_model


def tissue_record(cellMap, t):
    """
    Mean cell shape at time t: height, apical area and aspect ratio (height / sqrt(area))

    :param cellMap:
    :param t:
    :return:
    """
    alive = cellMap.face_df[cellMap.face_df["is_alive"] == 1]
    return {
        "time": t,
        "height": alive["height"].mean(),
        "area": alive["area"].mean(),
        "aspect_ratio": (alive["height"] / np.sqrt(alive["area"])).mean(),
        "vol": alive["vol"].mean(),
    }


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


def create_solver(cellMap, geom, energyContributions_model, history, manager):
    """
    EulerSolver with T1 transitions (periodic-safe T1s on periodic sheets)

    :param cellMap:
    :param geom:
    :param energyContributions_model:
    :param history:
    :param manager:
    :return:
    """
    periodic = cellMap.settings.get("boundaries") is not None
    if periodic:
        manager.append(periodic_reconnect, geom=geom)

    return EulerSolver(cellMap,
                       geom,
                       energyContributions_model,
                       history=history,
                       manager=manager,
                       auto_reconnect=not periodic)


def run_until(solver, endTime, dt):
    """
    Advance the solver from its current time (solver.prev_t, already computed) to endTime

    :param solver:
    :param endTime:
    :param dt:
    :return:
    """
    solver.prev_t += dt
    solver.solve(tf=endTime, dt=dt)


def mean_vertex_speed(cellMap, previousPositions, elapsedTime):
    """
    Mean distance moved by the vertices per unit time (minimum image on periodic sheets)

    :param cellMap:
    :param previousPositions: vert_df[coords] at the beginning of the interval
    :param elapsedTime:
    :return:
    """
    # T1 transitions can add or remove vertices: compare the ones present at both times
    common = cellMap.vert_df.index.intersection(previousPositions.index)
    displacement = cellMap.vert_df.loc[common, cellMap.coords] - previousPositions.loc[common]
    for coord, (low, high) in (cellMap.settings.get("boundaries") or {}).items():
        period = high - low
        displacement[coord] -= period * np.round(displacement[coord] / period)

    return np.linalg.norm(displacement.to_numpy(), axis=1).mean() / elapsedTime


def solve_until_relaxed(cellMap, geom, energyContributions_model, tol=1e-3, maxTime=100):
    """
    Simulate until the vertices don't move much (relaxed tissue)

    :param cellMap:
    :param geom:
    :param energyContributions_model:
    :param tol: relaxed when the mean vertex speed (distance per minute) is below tol
    :param maxTime: maximum simulated time (minutes)
    :return: [cellMap, geom, energyContributions_model, history, relaxTimecourse]
        relaxTimecourse: DataFrame with the mean vertex speed, cell height, area and aspect
        ratio every 'record_every' minutes
    """
    dt = cellMap.settings["dt"]
    checkEvery = cellMap.settings["record_every"]

    ## Init history object, recording every checkEvery minutes
    history_cellMap = History(cellMap, save_every=round(checkEvery / dt), dt=1)

    ## Init solver
    manager = EventManager("face")
    solver = create_solver(cellMap, geom, energyContributions_model, history_cellMap, manager)

    ## Run until the vertices stop moving
    t = 0.0
    speed = np.inf
    geom.update_all(cellMap)
    relaxTimecourse = [{**tissue_record(cellMap, t), "speed": np.nan}]
    while t < maxTime and speed > tol:
        previousPositions = cellMap.vert_df[cellMap.coords].copy()
        run_until(solver, t + checkEvery, dt)
        t += checkEvery
        speed = mean_vertex_speed(cellMap, previousPositions, checkEvery)
        relaxTimecourse.append({**tissue_record(cellMap, t), "speed": speed})

    print(f"Relaxation stopped at t = {t:g} min, mean vertex speed = {speed:.2e}")

    ## Deep copy to return it and being able to modify maintaining the previous one
    cellMap_new = copy.deepcopy(cellMap)
    history_new = copy.deepcopy(history_cellMap)

    return [cellMap_new, geom, energyContributions_model, history_new, pd.DataFrame(relaxTimecourse)]


def wound_healing_event(sheet, manager, tAblation, woundTimecourse):
    """
    EventManager behaviour run after every Euler step: updates the wound tension and
    records the wound every 'record_every' minutes

    :param sheet:
    :param manager:
    :param tAblation: time of the ablation (minutes)
    :param woundTimecourse: list where the wound records are appended
    :return:
    """
    dt = sheet.settings["dt"]
    numStep = manager.clock + 1  # manager.clock counts the steps already finished
    t = tAblation + numStep * dt

    wound.update_wound_tension(sheet, t)
    if numStep % round(sheet.settings["record_every"] / dt) == 0:
        woundTimecourse.append(wound.wound_record(sheet, t))

    manager.append(wound_healing_event, tAblation=tAblation, woundTimecourse=woundTimecourse)


def solve_with_ablation(cellMap, geom, energyContributions_model):
    """
    Perform an ablation and simulate until tend.
    The ablation happens at TInitAblation, so the wound contractility follows the same
    time profile as in the 3D model

    :param cellMap: relaxed tissue
    :param geom:
    :param energyContributions_model:
    :return: [cellMap, geom, energyContributions_model, history, woundTimecourse]
        woundTimecourse: DataFrame with the wound measures every 'record_every' minutes,
        to be summarised with wound.wound_features (recoil, max wound area, healing rate)
    """
    dt = cellMap.settings["dt"]
    tAblation = cellMap.settings["TInitAblation"]
    endTime = cellMap.settings["tend"]

    ## Ablation: the central cells become a single debris cell
    ablatedCells = wound.central_faces(cellMap, cellMap.settings["n_ablated"])
    wound.ablate(cellMap, ablatedCells, geom)
    wound.update_wound_tension(cellMap, tAblation)
    woundTimecourse = [wound.wound_record(cellMap, tAblation)]

    ## Init history object, recording every 'record_every' minutes
    history_cellMap = History(cellMap, save_every=round(cellMap.settings["record_every"] / dt), dt=1)

    ## Init solver, with the wound protocol executed after every step
    manager = EventManager("face")
    manager.append(wound_healing_event, tAblation=tAblation, woundTimecourse=woundTimecourse)
    solver = create_solver(cellMap, geom, energyContributions_model, history_cellMap, manager)
    solver.prev_t = tAblation

    ## Run the solver
    run_until(solver, endTime, dt)

    ## Deep copy to return it and being able to modify maintaining the previous one
    cellMap_new = copy.deepcopy(cellMap)
    history_new = copy.deepcopy(history_cellMap)

    return [cellMap_new, geom, energyContributions_model, history_new, pd.DataFrame(woundTimecourse)]


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
