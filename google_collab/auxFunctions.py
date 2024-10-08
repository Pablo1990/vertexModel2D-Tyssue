import pathlib
import random

import matplotlib.pylab as plt
from tyssue.draw import sheet_view
from tyssue.io import obj


def line_tension_range(cellmap, lower_line_tension, higher_line_tension):
    """
    Randomly assign line tension values between lower_line_tension and higher_line_tension
    :param cellmap:
    :param lower_line_tension:
    :param higher_line_tension:
    :return:
    """
    for edge in range(len(cellmap.edge_df)):
        newValue = random.uniform(lower_line_tension, higher_line_tension)
        cellmap.edge_df.loc[edge, 'line_tension'] = newValue
    return cellmap


def create_frames(
        history,
        output,
        num_frames=None,
        interval=None,
        draw_func=None,
        margin=5,
        **draw_kwds,
):
    """
    Creates a set of png frames of the recorded history.
    :param history:  a :class:`tyssue.History` object
    :param output:  path to the output directory
    :param num_frames:  int, the number of frames in the gif
    :param interval:    tuples, define begin and end frame of the gif
    :param draw_func:   a drawing function. This function must take a `sheet` object as first argument and return a
    `fig, ax` pair. Defaults to quick_edge_draw(aka sheet_view with quick mode)
    :param margin:  int, the graph margins in percents, default 5. If margin is -1, let the draw function decide
    :param draw_kwds:   are passed to the drawing function
    :return:
    """

    graph_dir = pathlib.Path(output)
    graph_dir.mkdir(parents=True, exist_ok=True)

    x, y = coords = draw_kwds.get("coords", history.sheet.coords[:2])
    sheet0 = history.retrieve(0)
    bounds = sheet0.vert_df[coords].describe().loc[["min", "max"]]
    delta = (bounds.loc["max"] - bounds.loc["min"]).max()
    margin = delta * margin / 100
    xlim = bounds.loc["min", x] - margin, bounds.loc["max", x] + margin
    ylim = bounds.loc["min", y] - margin, bounds.loc["max", y] + margin

    if interval is None:
        start, stop = None, None
    else:
        start, stop = interval[0], interval[1]

    for i, (t, sheet) in enumerate(history.browse(start, stop, num_frames)):
        try:
            fig, ax = sheet_view(sheet, **draw_kwds)
            fig.set_size_inches(20, 20)

            if isinstance(ax, plt.Axes) and margin >= 0:
                ax.set(xlim=xlim, ylim=ylim)

            plt.axis('off')
            fig.savefig(graph_dir / f"movie_{i:04d}.png")
        except Exception as e:
            print("Droped frame {i}")
            print(e)

        plt.close()


def exportToMesh(history, dir):
    """
    Exporting each timepoint to mesh
    :param history:
    :param dir:
    :return:
    """

    for i, (t, sheet) in enumerate(history.browse(None, None, None)):
        obj.save_splitted_cells(dir + '/junctions_' + str(t) + '.obj', sheet, epsilon=0.001)
