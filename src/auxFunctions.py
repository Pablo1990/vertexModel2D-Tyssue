import pathlib

import matplotlib.pyplot as plt
import meshio
import numpy as np

from tyssue.draw import sheet_view


def line_tension_range(cellmap, lower_line_tension, higher_line_tension):
    """
    Randomly assign line tension values between lower_line_tension and higher_line_tension
    :param cellmap:
    :param lower_line_tension:
    :param higher_line_tension:
    :return:
    """
    cellmap.edge_df["line_tension"] = np.random.uniform(
        lower_line_tension, higher_line_tension, size=cellmap.Ne
    )
    return cellmap


def length_elasticity_range(cellmap, lower_value, higher_value):
    cellmap.edge_df["length_elasticity"] = np.random.uniform(
        lower_value, higher_value, size=cellmap.Ne
    )
    return cellmap


def create_frames(
    history,
    output,
    num_frames=None,
    interval=None,
    margin=0,
    **draw_kwds,
):
    """Creates a set of png frames of the recorded history.

    Parameters
    ----------
    history : a :class:`tyssue.History` object
    output : path to the output directory
    num_frames : int, the number of frames (default: one per recorded time step)
    interval : tuple of int, (first, last) time-point *index* to draw
    margin : int, the graph margins in percents, default 0
         if margin is -1, let the draw function decide
    **draw_kwds are passed to the drawing function
    """
    graph_dir = pathlib.Path(output)
    graph_dir.mkdir(parents=True, exist_ok=True)

    x, y = coords = draw_kwds.get("coords", history.sheet.coords[:2])
    sheet0 = history.retrieve(history.time_stamps[0])
    bounds = sheet0.vert_df[coords].describe().loc[["min", "max"]]
    delta = (bounds.loc["max"] - bounds.loc["min"]).max()
    margin = delta * margin / 100
    xlim = bounds.loc["min", x] - margin, bounds.loc["max", x] + margin
    ylim = bounds.loc["min", y] - margin, bounds.loc["max", y] + margin

    start, stop = (0, None) if interval is None else interval

    # History.browse works again in vivatyssue: it yields (time, sheet) pairs
    # using the real recorded time stamps, whatever dt was.
    for i, (t, sheet) in enumerate(
        history.browse(start=start, stop=stop, size=num_frames)
    ):
        try:
            fig, ax = sheet_view(sheet, **draw_kwds)
            fig.set_size_inches(20, 20)
            # if isinstance(ax, plt.Axes) and margin >= 0:
            #     ax.set(xlim=xlim, ylim=ylim)
            plt.axis("off")
            fig.savefig(graph_dir / f"movie_{i:04d}.png")
        except Exception as e:
            print(f"Dropped frame {i} (t={t}): {e}")
        finally:
            plt.close("all")


def save_splitted_cells(fname, sheet, epsilon=0.001):
    """Saves the sheet as a triangular mesh where each cell is shrunk towards
    its centre by `epsilon`, so neighbouring cells do not share vertices.

    Same output as `tyssue.io.obj.save_splitted_cells`, but written with
    meshio (a core vivatyssue dependency) instead of vispy (`tyssue[viz]`).
    """
    coords = sheet.coords
    up_srce = sheet.upcast_srce(sheet.vert_df[coords]).to_numpy()
    up_trgt = sheet.upcast_trgt(sheet.vert_df[coords]).to_numpy()
    up_face = sheet.upcast_face(sheet.face_df[coords]).to_numpy()
    up_srce = (up_srce - up_face) * (1 - epsilon) + up_face
    up_trgt = (up_trgt - up_face) * (1 - epsilon) + up_face

    points = np.vstack([sheet.face_df[coords].to_numpy(), up_srce, up_trgt])
    if points.shape[1] == 2:  # OBJ needs 3D points
        points = np.hstack([points, np.zeros((points.shape[0], 1))])

    Ne, Nf = sheet.Ne, sheet.Nf
    face_pos = sheet.face_df.index.get_indexer(sheet.edge_df["face"])
    triangles = np.vstack(
        [face_pos, np.arange(Ne) + Nf, np.arange(Ne) + Ne + Nf]
    ).T
    meshio.write_points_cells(str(fname), points, [("triangle", triangles)])


def exportToMesh(history, dir):
    """
    Exporting each timepoint to mesh
    :param history:
    :param dir:
    :return:
    """
    out_dir = pathlib.Path(dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t, sheet in history:
        try:
            save_splitted_cells(out_dir / f"junctions_{int(t)}.obj", sheet, epsilon=0.001)
        except Exception as e:
            print(f"Error at time {t}: {e}")
