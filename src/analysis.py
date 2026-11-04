"""
Features of the wound healing simulations by aspect ratio (AR) and their plots, following
pyVertexModel util.utils.plot_figure_with_line: boxplot + points per AR, and a curve fitted to
all the points (R^2 computed on the average by AR).

Expected layout (main_wound_healing.py):
    <model_folder>/<AR>/required_purse_string.json  height, recoil, purse_string_strength, ...
    <model_folder>/<AR>/calibrated_parameters.json  calibrated lambdaS
    <model_folder>/<AR>/calibration.csv             features of each calibration evaluation
    <model_folder>/<AR>/wound_timecourse.csv        wound area over time
"""
import json
import os
import warnings
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import OptimizeWarning, curve_fit

import src.wound as wound
from src.vertexModel import REQUIRED_PURSE_STRING_FILE

AR = "AR"
WOUND_AREA_TIME = 6.0  # minutes after ablation
# Before REQUIRED_PURSE_STRING_FILE, one CSV per model folder (rewritten by each run)
LEGACY_PURSE_STRING_FILE = "required_purse_string.csv"


@dataclass(frozen=True)
class Feature:
    column: str
    label: str
    log_y: bool = False


# Features plotted against AR, in the order of the figures
FEATURES = [
    Feature("lambdaS", r"$\lambda_{S}$", log_y=True),
    Feature("recoil", "Recoil velocity (t=0.1)"),
    Feature("wound_area_6", "Wound area at 6 min. (%)"),
    Feature("healing_rate", r"Healing rate (min$^{-1}$)"),
    Feature("edge_recoil", r"Edge recoil (s$^{-1}$)"),
    Feature("purse_string_strength", "Purse string strength (t=0.1)"),
]


def power_law(x, a, b):
    """y = a * x^b, as pyVertexModel purse_string_strength_curve"""
    return a * np.power(x, b)


def r2(y, y_fit):
    """Coefficient of determination"""
    y, y_fit = np.asarray(y, dtype=float), np.asarray(y_fit, dtype=float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - np.sum((y - y_fit) ** 2) / ss_tot if ss_tot > 0 else np.nan


def is_ar_folder(path):
    """A simulation folder is named after its AR (e.g. '7.5')"""
    try:
        float(os.path.basename(path))
    except ValueError:
        return False
    return os.path.isdir(path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def calibrated_features(ar_folder, lambda_s):
    """
    Features of the calibration evaluation run with the calibrated lambdaS (edge recoil is only
    computed there)
    """
    calibration_file = os.path.join(ar_folder, "calibration.csv")
    if lambda_s is None or not os.path.exists(calibration_file):
        return {}
    calibration = pd.read_csv(calibration_file)
    best = calibration.loc[(calibration["lambdaS"] - lambda_s).abs().idxmin()]
    return {name: float(best[name]) for name in ("edge_recoil", "edge_recoil_K") if name in best}


def simulation_features(ar_folder):
    """Features of one simulation folder; missing outputs are left as NaN"""
    features = {AR: float(os.path.basename(ar_folder))}

    parameters_file = os.path.join(ar_folder, "calibrated_parameters.json")
    lambda_s = read_json(parameters_file).get("lambdaS") if os.path.exists(parameters_file) else None
    features["lambdaS"] = lambda_s
    features.update(calibrated_features(ar_folder, lambda_s))

    purse_string_file = os.path.join(ar_folder, REQUIRED_PURSE_STRING_FILE)
    if os.path.exists(purse_string_file):
        purse_string = read_json(purse_string_file)
        purse_string.pop("height", None)  # already the AR
        features.update(purse_string)

    timecourse_file = os.path.join(ar_folder, "wound_timecourse.csv")
    if os.path.exists(timecourse_file):
        timecourse = pd.read_csv(timecourse_file)
        features["wound_area_6"] = wound.relative_wound_area(timecourse, WOUND_AREA_TIME)
        features.update({f"timecourse_{k}": v for k, v in wound.wound_features(timecourse).items()})
        features["healing_rate"] = features.pop("timecourse_healing_rate")
    return features


def ar_folders(model_folder):
    """Simulation folders of a model folder"""
    folders = (os.path.join(model_folder, name) for name in os.listdir(model_folder))
    return [folder for folder in folders if is_ar_folder(folder)]


def legacy_purse_strings(model_folder):
    """Purse string results of the model-level CSV, for the ARs without their own file"""
    legacy_file = os.path.join(model_folder, LEGACY_PURSE_STRING_FILE)
    if not os.path.exists(legacy_file):
        return pd.DataFrame(columns=[AR])
    return pd.read_csv(legacy_file).rename(columns={"height": AR})


def collect_features(model_folder):
    """
    One row per simulation (AR folder) of a model folder
    :return: DataFrame sorted by AR
    """
    rows = [simulation_features(folder) for folder in ar_folders(model_folder)]
    features = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[AR])

    legacy = legacy_purse_strings(model_folder)
    if not legacy.empty:  # old results: one simulation per AR, missing values taken from the CSV
        legacy = legacy.drop_duplicates(AR, keep="last").set_index(AR)
        features = features.set_index(AR).combine_first(legacy).reset_index()
    return features.sort_values(AR).reset_index(drop=True)


def fit_power_law(x, y, log_space=False):
    """
    Fit y = a x^b; None with fewer than 3 different ARs or if it does not converge
    :param log_space: fit log(y) = log(a) + b log(x), for positive features spanning orders of
        magnitude (otherwise the largest values dominate the least squares)
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(np.unique(x)) < 3:
        return None
    if log_space:
        if np.any(y <= 0):
            return None
        b, log_a = np.polyfit(np.log(x), np.log(y), 1)
        return np.array([np.exp(log_a), b])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)  # covariance not estimated: not used
            params, _ = curve_fit(power_law, x, y, p0=(float(np.mean(y)), 0.0), maxfev=100000)
    except (RuntimeError, ValueError):
        return None
    return params


def plot_feature_by_ar(features, feature, output_folder, x_label="Aspect ratio (AR)"):
    """
    Boxplot + points by AR, and the power law fitted to all the points
    :return: path of the figure, or None if there is no data
    """
    if feature.column not in features:
        return None
    data = features[[AR, feature.column]].dropna()
    if data.empty:
        return None
    ar_order = np.sort(data[AR].unique())

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=AR, y=feature.column, data=data, order=ar_order, hue=AR, hue_order=ar_order,
                     palette="vlag", legend=False, whis=[0, 100], width=.6, showmeans=True,
                     meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"})
    sns.stripplot(x=AR, y=feature.column, data=data, order=ar_order, size=4, color=".3", ax=ax)

    mean_by_ar = data.groupby(AR)[feature.column].mean().reindex(ar_order).to_numpy()
    params = fit_power_law(data[AR], data[feature.column], log_space=feature.log_y)
    if params is not None:
        mean_fit = power_law(ar_order, *params)
        fit_r2 = r2(np.log(mean_by_ar), np.log(mean_fit)) if feature.log_y else r2(mean_by_ar, mean_fit)
        label = f"$y = {params[0]:.2e} \\cdot x^{{{params[1]:.2f}}}$ - R$^2$ = {fit_r2:.2f}"
        # Smooth curve on the categorical axis: positions interpolated in log(AR)
        ar_curve = np.geomspace(ar_order[0], ar_order[-1], 200)
        x_curve = np.interp(np.log(ar_curve), np.log(ar_order), np.arange(len(ar_order)))
        ax.plot(x_curve, power_law(ar_curve, *params), color="black", linewidth=2, label=label)
        ax.legend()
    if feature.log_y:
        ax.set_yscale("log")

    ax.set_xticks(np.arange(len(ar_order)))
    ax.set_xticklabels([f"{ar:g}" for ar in ar_order], rotation=45, fontsize=20, fontweight="bold")
    plt.yticks(fontsize=20, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=20, fontweight="bold")
    ax.set_ylabel(feature.label, fontsize=20, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    figure_file = os.path.join(output_folder, f"boxplot_{AR}_{feature.column}.png")
    plt.savefig(figure_file)
    plt.close()
    return figure_file


def plot_model_folder(model_folder, output_folder=None):
    """
    Collect the features of a model folder, save them (per simulation and averaged by AR) and plot
    every feature in FEATURES against AR
    :return: (features DataFrame, list of figure paths)
    """
    output_folder = output_folder or os.path.join(model_folder, "plots")
    os.makedirs(output_folder, exist_ok=True)

    features = collect_features(model_folder)
    features.to_csv(os.path.join(output_folder, "features.csv"), index=False)
    features.groupby(AR).mean(numeric_only=True).to_csv(os.path.join(output_folder, "features_mean_by_AR.csv"))

    figures = [plot_feature_by_ar(features, feature, output_folder) for feature in FEATURES]
    return features, [figure for figure in figures if figure is not None]
