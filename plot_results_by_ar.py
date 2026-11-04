"""
Plot the wound healing features against the aspect ratio (AR) for one or more model folders
(results/wound_healing/<...>/<model>), as pyVertexModel's plot_figure_with_line.

    python plot_results_by_ar.py                                   # every model folder found
    python plot_results_by_ar.py results/wound_healing/same_vol_wound_area/effective_2d

Figures and CSVs go to <model folder>/plots/.
"""
import argparse
import os

from src.analysis import ar_folders, plot_model_folder

RESULTS_FOLDER = os.path.join("results", "wound_healing")


def find_model_folders(root):
    """Folders with simulation (AR) subfolders, i.e. written by main_wound_healing.py"""
    return sorted(folder for folder, _, _ in os.walk(root) if ar_folders(folder))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_folders", nargs="*", help=f"default: every model folder in {RESULTS_FOLDER}")
    args = parser.parse_args()

    for model_folder in args.model_folders or find_model_folders(RESULTS_FOLDER):
        features, figures = plot_model_folder(model_folder)
        print(f"{model_folder}: {len(features)} simulations, {len(figures)} figures")


if __name__ == "__main__":
    main()
