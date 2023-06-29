from tyssue import config
import auxFunctions
import inputMechanicalParameters

def initialize(area_elasticity, prefered_area, lower_line_tension, higher_line_tension):
    ## Defining energy contributions
    # https://tyssue.readthedocs.io/en/latest/_modules/tyssue/dynamics/effectors.html
    model_init = model_factory([    
        effectors.FaceAreaElasticity,
        effectors.LineTension,
        #effectors.LengthElasticity,
        effectors.PerimeterElasticity
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
    cellmap_init = Sheet.planar_sheet_2d('tissue', numCellRows, numCellRows, 1, 1, noise=noiseCellShape)
    cellmap_init.remove(cellmap_init.cut_out([[1, numCellRows], [1, numCellRows]]), trim_borders=True)
    cellmap_init.reset_index()
    cellmap_init.reset_topo()

    ## Definition of the geometry of the sheet
    # PlanarGeometry: Geometry methods for 2D planar cell arangements
    # SheetGeometry: Geometry definitions for 2D sheets in 3D
    # BulkGeometry: Geometry functions for 3D cell arangements
    geom_init  = PlanarGeometry

    # Update geometry with the patch
    geom_init.update_all(cellmap_init)

    # Visualize the sheet
    fig, ax = sheet_view(cellmap_init, mode="quick")

    ## Connect cells with energy contributions
    cellmap_init.update_specs(model_init.specs)

    ## Init history object
    # It gives a warning when using: ', extra_cols={"edge":["dx", "dy"]}'
    #TODO: WHAT TO USE INSTEAD?
    history_init = History(cellmap_init)

    return [cellmap_init]


def solve(cellmap, history_original, endTime):

    ## Find the minima
    solver1 = EulerSolver(cellmap, geom_original, model_original, history=history_original, auto_reconnect=True)
    res1 = solver1.solve(tf=endTime, dt=0.05)

    ## Deep copy to return it and being able to modify maintaining the previous one
    cellmap_new = copy.deepcopy(cellmap)
    history_new = copy.deepcopy(history_original)

    return [cellmap_new, geom_new, model_new, history_new]