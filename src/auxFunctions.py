import pathlib
import random
import numpy as np

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

    # Replace the loop that uses 'browse' with this manual approach
    for i, t in enumerate(np.linspace(start if start else 0, stop if stop else history.time,
                                      num_frames if num_frames else len(history.time_stamps))):
        try:
            # Manually retrieve each sheet for the corresponding time step
            sheet = history.retrieve(int(t))  # retrieve from history at time 't'

            # Now apply the sheet view function as before
            fig, ax = sheet_view(sheet, **draw_kwds)
            fig.set_size_inches(20, 20)

            # Set the xlim and ylim for margins, as in the original code
            if isinstance(ax, plt.Axes) and margin >= 0:
                ax.set(xlim=xlim, ylim=ylim)

            plt.axis('off')
            fig.savefig(graph_dir / f"movie_{i:04d}.png")
        except Exception as e:
            print(f"Dropped frame {i}")
            print(e)

        plt.close()


def exportToMesh(history, dir):
    """
    Exporting each timepoint to mesh
    :param history:
    :param dir:
    :return:
    """

    # Replace the loop that uses 'browse' with this manual approach
    for i, t in enumerate(np.linspace(0, history.time, len(history.time_stamps))):
        try:
            # Manually retrieve each sheet for the corresponding time step
            sheet = history.retrieve(int(t))  # retrieve from history at time 't'

            # Save the split cells data using the provided function
            obj.save_splitted_cells(dir + f'/junctions_{int(t)}.obj', sheet, epsilon=0.001)

        except Exception as e:
            print(f"Error at time {t}: {e}")

