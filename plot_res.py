# -*- coding: utf-8 -*-

"""
Plot QD-TD scatter plots.

Two subplots:
(a) G2
(b) G3

Visual encoding:
1. Method
   ME / MS-ME       -> red
   LPSI / MS-LPSI   -> blue
   ILP / MAPS       -> green

2. Problem
   Single source    -> light color
   Multiple sources -> dark color

3. Scenario
   Noise-free                 -> triangle (^)
   Probabilistic propagation  -> star (*)
   Observation error          -> circle (o)

Legend:
Method   : 3 items
Scenario : 3 items
Problem  : 2 items
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. Matplotlib settings
# ============================================================

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. Method definitions
# ============================================================

methods_base = ["ME", "LPSI", "ILP"]

# Each baseline method is immediately followed by
# its multiscale counterpart
methods = [
    "ME", "MS-ME",
    "LPSI", "MS-LPSI",
    "ILP", "MAPS"
]

method_to_idx = {
    method: i for i, method in enumerate(methods)
}

print("method_to_idx:", method_to_idx)


# ============================================================
# 3. Color definitions
# ============================================================

# Light color  -> single-source problem
# Dark color   -> multiple-source problem

method_problem_to_color = {

    # ME family: red
    "ME": {
        "single": "#FEB7B0",
        "multiple": "#F9220A"
    },

    # LPSI family: blue
    "LPSI": {
        "single": "#9AD7FF",
        "multiple": "#0485FD"
    },

    # ILP / MAPS family: green
    "ILP": {
        "single": "#9AFEC4",
        "multiple": "#03A747"
    }
}


# Dark representative colors for the Method legend
method_legend_color = {
    "ME": "#F9220A",
    "LPSI": "#0485FD",
    "ILP": "#03A747"
}


# ============================================================
# 4. Scenario marker definitions
# ============================================================

scenario_to_marker = {
    "Noise-free": "^",
    "Probabilistic Propagation": "*",
    "Observation Error": "o"
}


# ============================================================
# 5. Construct dataframe
# ============================================================

rows = []


def add_block(
        dataset,
        scenario,
        single_precisions,
        single_times,
        mult_f1s,
        mult_times
):
    """
    Calculate QD, TD, and EI for:

        ME     -> MS-ME
        LPSI   -> MS-LPSI
        ILP    -> MAPS

    QD = (Q_MS - Q_base) / Q_base
    TD = (T_MS - T_base) / T_base

    EI = -TD / |QD| - 1
    """

    for method_base in methods_base:

        idx = method_to_idx[method_base]

        # ====================================================
        # Single-source
        # ====================================================

        single_quality_difference = (
            single_precisions[idx + 1]
            / single_precisions[idx]
        ) - 1

        single_time_difference = (
            single_times[idx + 1]
            / single_times[idx]
        ) - 1


        # EI
        if single_quality_difference == 0:

            if single_time_difference < 0:
                single_ei = np.inf

            elif single_time_difference > 0:
                single_ei = -np.inf

            else:
                single_ei = np.nan

        else:

            single_ei = (
                -single_time_difference
                / abs(single_quality_difference)
                - 1
            )


        rows.append({
            "dataset": dataset,
            "scenario": scenario,
            "problem": "single",
            "method_base": method_base,
            "quality_improvement": single_quality_difference,
            "time_improvement": single_time_difference,
            "EI": single_ei
        })


        # ====================================================
        # Multiple-source
        # ====================================================

        multiple_quality_difference = (
            mult_f1s[idx + 1]
            / mult_f1s[idx]
        ) - 1

        multiple_time_difference = (
            mult_times[idx + 1]
            / mult_times[idx]
        ) - 1


        # EI
        if multiple_quality_difference == 0:

            if multiple_time_difference < 0:
                multiple_ei = np.inf

            elif multiple_time_difference > 0:
                multiple_ei = -np.inf

            else:
                multiple_ei = np.nan

        else:

            multiple_ei = (
                -multiple_time_difference
                / abs(multiple_quality_difference)
                - 1
            )


        rows.append({
            "dataset": dataset,
            "scenario": scenario,
            "problem": "multiple",
            "method_base": method_base,
            "quality_improvement": multiple_quality_difference,
            "time_improvement": multiple_time_difference,
            "EI": multiple_ei
        })


# ============================================================
# 6. Experimental data
# ============================================================


# ------------------------------------------------------------
# G2: Noise-free
# ------------------------------------------------------------

add_block(
    "G2",
    "Noise-free",

    single_precisions=[
        100.00, 100.00,
        100.00, 100.00,
        100.00, 100.00
    ],

    single_times=[
        0.3594, 0.3681,
        0.5292, 0.6469,
        0.7480, 0.8838
    ],

    mult_f1s=[
        82.13, 67.05,
        82.13, 67.05,
        82.13, 67.05
    ],

    mult_times=[
        0.2506, 0.3109,
        0.3602, 0.3882,
        1.0652, 0.5879
    ]
)


# ------------------------------------------------------------
# G2: Probabilistic Propagation
# ------------------------------------------------------------

add_block(
    "G2",
    "Probabilistic Propagation",

    single_precisions=[
        100.00, 100.00,
        100.00, 100.00,
        100.00, 100.00
    ],

    single_times=[
        0.3426, 0.4222,
        0.5173, 0.7197,
        0.7314, 0.8302
    ],

    mult_f1s=[
        95.77, 71.39,
        95.77, 71.39,
        95.77, 71.39
    ],

    mult_times=[
        0.2585, 0.2796,
        0.3601, 0.4200,
        0.6399, 0.4333
    ]
)


# ------------------------------------------------------------
# G2: Observation Error
# ------------------------------------------------------------

add_block(
    "G2",
    "Observation Error",

    single_precisions=[
        50.26, 13.47,
        51.81, 12.95,
        67.88, 32.54
    ],

    single_times=[
        0.4180, 0.4608,
        0.5464, 0.8670,
        2.4400, 0.6459
    ],

    mult_f1s=[
        34.27, 26.30,
        49.45, 20.67,
        63.00, 31.14
    ],

    mult_times=[
        0.2564, 0.2046,
        0.3638, 0.4013,
        2.1175, 0.4629
    ]
)


# ------------------------------------------------------------
# G3: Noise-free
# ------------------------------------------------------------

add_block(
    "G3",
    "Noise-free",

    single_precisions=[
        100.00, 100.00,
        100.00, 100.00,
        100.00, 100.00
    ],

    single_times=[
        13.6602, 11.0153,
        20.0177, 16.9239,
        73.4500, 30.9897
    ],

    mult_f1s=[
        90.85, 90.29,
        90.85, 90.29,
        90.85, 90.29
    ],

    mult_times=[
        1.2463, 0.8923,
        2.9568, 1.0669,
        8.5357, 1.1231
    ]
)


# ------------------------------------------------------------
# G3: Probabilistic Propagation
# ------------------------------------------------------------

add_block(
    "G3",
    "Probabilistic Propagation",

    single_precisions=[
        100.00, 100.00,
        100.00, 100.00,
        100.00, 100.00
    ],

    single_times=[
        15.5577, 16.7458,
        13.8099, 16.9787,
        23.5590, 14.8378
    ],

    mult_f1s=[
        99.32, 98.30,
        99.32, 98.30,
        99.46, 98.20
    ],

    mult_times=[
        1.0783, 1.0869,
        1.1735, 1.2674,
        2.4123, 1.1807
    ]
)


# ------------------------------------------------------------
# G3: Observation Error
# ------------------------------------------------------------

add_block(
    "G3",
    "Observation Error",

    single_precisions=[
        27.80, 2.32,
        34.70, 10.72,
        54.12, 41.76
    ],

    single_times=[
        16.4999, 14.4339,
        19.0498, 19.6285,
        275.2930, 121.4809
    ],

    mult_f1s=[
        7.57, 9.13,
        26.60, 18.00,
        40.94, 32.68
    ],

    mult_times=[
        1.3653, 1.0683,
        1.9394, 1.2751,
        22.9977, 3.1484
    ]
)


# ============================================================
# 7. Convert to dataframe
# ============================================================

df = pd.DataFrame(rows)

print(df)


# ============================================================
# 8. Coordinate compression function
# ============================================================

def compress_coordinate(x, y, limit=100):
    """
    If either |x| or |y| exceeds `limit`,
    proportionally shrink both x and y.

    This preserves the direction / slope of the point
    relative to the origin.
    """

    x = float(x)
    y = float(y)

    max_abs = max(abs(x), abs(y))

    if max_abs > limit:

        ratio = max_abs / limit

        x = x / ratio
        y = y / ratio

    return x, y


# ============================================================
# 9. Draw scatter plots
# ============================================================

# Preserve the original figure ratio
fig, axs = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

axs = axs.flatten()

datasets = [
    "G2",
    "G3"
]


for i, ax in enumerate(axs):

    dataset = datasets[i]

    df_dataset = df[
        df["dataset"] == dataset
    ]


    # --------------------------------------------------------
    # Plot data points
    # --------------------------------------------------------

    for scenario in [
        "Noise-free",
        "Probabilistic Propagation",
        "Observation Error"
    ]:

        marker = scenario_to_marker[scenario]

        df_scenario = df_dataset[
            df_dataset["scenario"] == scenario
        ]


        for problem in [
            "single",
            "multiple"
        ]:

            df_problem = df_scenario[
                df_scenario["problem"] == problem
            ]


            for method_base in methods_base:

                df_method = df_problem[
                    df_problem["method_base"]
                    == method_base
                ]

                if df_method.empty:
                    continue


                # QD (%)
                x = (
                    df_method.iloc[0][
                        "quality_improvement"
                    ]
                    * 100
                )

                # TD (%)
                y = (
                    df_method.iloc[0][
                        "time_improvement"
                    ]
                    * 100
                )


                # Compress extreme values proportionally
                x, y = compress_coordinate(
                    x,
                    y,
                    limit=100
                )


                # Point color
                color = (
                    method_problem_to_color
                    [method_base]
                    [problem]
                )


                # Scatter point
                ax.scatter(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    s=80,
                    edgecolors="none",
                    zorder=3
                )


    # ========================================================
    # Reference lines
    # ========================================================

    # y = 0
    ax.axhline(
        y=0,
        color="gray",
        linestyle="-",
        linewidth=0.7,
        zorder=1
    )

    # x = 0
    ax.axvline(
        x=0,
        color="gray",
        linestyle="-",
        linewidth=0.7,
        zorder=1
    )

    # y = x
    ax.plot(
        [-105, 105],
        [-105, 105],
        color="gray",
        linestyle="--",
        linewidth=0.8,
        zorder=1
    )


    # ========================================================
    # Axes
    # ========================================================

    ax.set_xlim(-105, 105)
    ax.set_ylim(-105, 105)

    ax.set_box_aspect(1)

    ax.set_xlabel(
        "Quality Difference (%)"
    )

    ax.set_ylabel(
        "Time Difference (%)"
    )


    # Grid
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.4,
        alpha=0.3
    )


    # ========================================================
    # Titles
    # ========================================================

    if dataset == "G2":

        ax.set_title(
            r"(a) $G_2$"
        )

    else:

        ax.set_title(
            r"(b) $G_3$"
        )


# ============================================================
# 10. Construct legends
# ============================================================


# ------------------------------------------------------------
# Legend 1: Method
# ------------------------------------------------------------

method_handles = [

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=7,
        markerfacecolor=method_legend_color["ME"],
        markeredgecolor="none",
        label="ME / MS-ME"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=7,
        markerfacecolor=method_legend_color["LPSI"],
        markeredgecolor="none",
        label="LPSI / MS-LPSI"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=7,
        markerfacecolor=method_legend_color["ILP"],
        markeredgecolor="none",
        label="ILP / MAPS"
    )
]


# ------------------------------------------------------------
# Legend 2: Scenario
#
# NF = Noise-free
# PP = Probabilistic propagation
# OE = Observation error
# ------------------------------------------------------------

scenario_handles = [

    Line2D(
        [0],
        [0],
        marker="^",
        linestyle="None",
        markersize=7,
        markerfacecolor="gray",
        markeredgecolor="gray",
        label="Noise-free"
    ),

    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markersize=10,
        markerfacecolor="gray",
        markeredgecolor="gray",
        label="Probabilistic propagation"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=7,
        markerfacecolor="gray",
        markeredgecolor="gray",
        label="Observation error"
    )
]


# ------------------------------------------------------------
# Legend 3: Problem
# ------------------------------------------------------------

problem_handles = [

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=7,
        markerfacecolor="#D3D3D3",
        markeredgecolor="none",
        label="Single source"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=7,
        markerfacecolor="#555555",
        markeredgecolor="none",
        label="Multiple sources"
    )
]


# ============================================================
# 11. Add three legends
# ============================================================

# All legend blocks use exactly the same x position
legend_x = 1.05


# ------------------------------------------------------------
# Method
# ------------------------------------------------------------

legend_method = axs[1].legend(
    handles=method_handles,
    title="Method",
    loc="upper left",
    bbox_to_anchor=(
        legend_x,
        1.00
    ),
    frameon=False,
    borderaxespad=0.0,
    handletextpad=0.6,
    labelspacing=0.5
)

legend_method.get_title().set_fontweight(
    "bold"
)

# Force title and contents to align left
legend_method._legend_box.align = "left"

# Preserve this legend when another legend is added
axs[1].add_artist(
    legend_method
)


# ------------------------------------------------------------
# Scenario
# ------------------------------------------------------------

legend_scenario = axs[1].legend(
    handles=scenario_handles,
    title="Scenario",
    loc="upper left",
    bbox_to_anchor=(
        legend_x,
        0.66
    ),
    frameon=False,
    borderaxespad=0.0,
    handletextpad=0.6,
    labelspacing=0.5
)

legend_scenario.get_title().set_fontweight(
    "bold"
)

# Force title and contents to align left
legend_scenario._legend_box.align = "left"

# Preserve this legend when another legend is added
axs[1].add_artist(
    legend_scenario
)


# ------------------------------------------------------------
# Problem
# ------------------------------------------------------------

legend_problem = axs[1].legend(
    handles=problem_handles,
    title="Problem",
    loc="upper left",
    bbox_to_anchor=(
        legend_x,
        0.30
    ),
    frameon=False,
    borderaxespad=0.0,
    handletextpad=0.6,
    labelspacing=0.5
)

legend_problem.get_title().set_fontweight(
    "bold"
)

# Force title and contents to align left
legend_problem._legend_box.align = "left"


# ============================================================
# 12. Layout
# ============================================================

# Preserve the original subplot proportions
plt.tight_layout(rect=[0, 0, 0.95, 1])


# ============================================================
# 13. Save figure
# ============================================================

output_dir = "scatter_plots"

os.makedirs(
    output_dir,
    exist_ok=True
)


output_path = os.path.join(
    output_dir,
    "exp_res_eff.pdf"
)


plt.savefig(
    output_path,
    dpi=600,
    bbox_inches="tight"
)


# ============================================================
# 14. Show figure
# ============================================================

plt.show()


print(
    "Figure saved to:",
    output_path
)