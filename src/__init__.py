import warnings

# vivatyssue moved ipyvolume/vispy into the optional `tyssue[viz]` extra and
# warns when they are missing. This project only uses the matplotlib drawing
# and meshio export, so those warnings are noise.
warnings.filterwarnings(
    "ignore", message=r".*tyssue\[viz\]", category=UserWarning
)
