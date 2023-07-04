from tyssue import History
from tyssue.draw.plt_draw import quick_edge_draw
from tyssue.io import obj
import matplotlib.pylab as plt
import pathlib

def line_tension_range(cellmap, lower_line_tension, higher_line_tension):
    for edge in range(len(cellmap.edge_df)):
        newValue = random.randrange(lower_line_tension, higher_line_tension)/100
        cellmap.edge_df['line_tension'][edge] = newValue         
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
    """Creates a set of png frames of the recorded history.
   
    Parameters
    ----------
    history : a :class:`tyssue.History` object
    output : path to the output directory
    num_frames : int, the number of frames in the gif
    interval : tuples, define begin and end frame of the gif
    draw_func : a drawing function
         this function must take a `sheet` object as first argument
         and return a `fig, ax` pair. Defaults to quick_edge_draw
         (aka sheet_view with quick mode)
    margin : int, the graph margins in percents, default 5
         if margin is -1, let the draw function decide
    **draw_kwds are passed to the drawing function
    """
    if draw_func is None:
        if draw_kwds.get("mode") in ("quick", None):
            draw_func = quick_edge_draw
        else:
            draw_func = sheet_view

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
            fig, ax = draw_func(sheet, **draw_kwds)
        except Exception as e:
            print("Droped frame {i}")

        if isinstance(ax, plt.Axes) and margin >= 0:
            ax.set(xlim=xlim, ylim=ylim)
        fig.savefig(graph_dir / f"movie_{i:04d}.png")
        plt.close(fig)

def exportToMesh(history, dir):
    """Exporting each timepoint to mesh"""
    for i, (t, sheet) in enumerate(history.browse(None, None, None)):
        obj.save_splitted_cells(dir + '/junctions_'+ str(t) +'.obj', sheet, epsilon=0.001)
