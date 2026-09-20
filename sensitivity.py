import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# 1. Basic settings
# ============================================================

LBL = ['SL', 'MLSL']
problem_list = ['single', 'multiple']
noise_list = ['PP', 'OE']

# Only keep ILP -> MAPS comparison
method = 'ILP'

# Parameter ranges stored in the JSON results
error_prop_list = np.arange(0.00, 0.51, 0.01)
prop_prob_list = np.arange(0.50, 1.01, 0.01)

quality_data = pd.DataFrame({
    'error_prop_list': error_prop_list,
    'prop_prob_list': prop_prob_list
})

time_data = pd.DataFrame({
    'error_prop_list': error_prop_list,
    'prop_prob_list': prop_prob_list
})


# ============================================================
# 2. Read experimental results
# ============================================================

for l in LBL:
    for p in problem_list:
        for n in noise_list:
            filename = f'results/{l}_{p}_{method}_{n}_L3.json'
            print(f'Reading: {filename}')
            with open(filename, 'r') as f:
                results = json.load(f)

            col_name = f'{l}_{p}_{method}_{n}'
            quality_data[col_name] = results['precision']
            time_data[col_name] = results['time']

print("\nQuality data:")
print(quality_data.head())

print("\nTime data:")
print(time_data.head())


# ============================================================
# 3. Compute QD and TD for MAPS relative to single-scale ILP
# ============================================================

def calculate_qd_td(problem, noise_type):
    sl_col = f'SL_{problem}_{method}_{noise_type}'
    ms_col = f'MLSL_{problem}_{method}_{noise_type}'

    sl_quality = quality_data[sl_col].astype(float)
    ms_quality = quality_data[ms_col].astype(float)

    sl_time = time_data[sl_col].astype(float)
    ms_time = time_data[ms_col].astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        qd = (ms_quality - sl_quality) / sl_quality
        td = (ms_time - sl_time) / sl_time

    return qd, td


# ============================================================
# 4. Parameter ranges used in the manuscript sensitivity plots
# ============================================================

def get_parameter_range(noise_type):
    if noise_type == 'PP':
        # Use P_r in [0.70, 1.00]
        indices = np.arange(20, 51)   # 0.70 to 1.00
        parameter = quality_data.loc[indices, 'prop_prob_list'].to_numpy()
        parameter_name = r'$P_r$'
        parameter_range_text = r'$P_r: 0.70 \rightarrow 1.00$'
    else:
        # Use P_e in [0.00, 0.30]
        indices = np.arange(0, 31)    # 0.00 to 0.30
        parameter = quality_data.loc[indices, 'error_prop_list'].to_numpy()
        parameter_name = r'$P_e$'
        parameter_range_text = r'$P_e: 0.00 \rightarrow 0.30$'

    return indices, parameter, parameter_name, parameter_range_text


# ============================================================
# 5. Draw one MAPS trajectory panel
# ============================================================

def plot_maps_trajectory(ax, problem, noise_type, panel_label):
    qd, td = calculate_qd_td(problem, noise_type)
    indices, parameter, parameter_name, parameter_range_text = get_parameter_range(noise_type)

    qd_plot = qd.loc[indices].to_numpy(dtype=float)
    td_plot = td.loc[indices].to_numpy(dtype=float)
    param_plot = parameter.copy()

    valid = np.isfinite(qd_plot) & np.isfinite(td_plot)
    qd_plot = qd_plot[valid]
    td_plot = td_plot[valid]
    param_plot = param_plot[valid]

    # --------------------------------------------------------
    # Base trajectory line
    # --------------------------------------------------------
    ax.plot(
        qd_plot,
        td_plot,
        color='gray',
        linewidth=1.5,
        alpha=0.8,
        zorder=1
    )

    # --------------------------------------------------------
    # Scatter points with color gradient
    # --------------------------------------------------------
    norm = Normalize(vmin=param_plot.min(), vmax=param_plot.max())
    scatter = ax.scatter(
        qd_plot,
        td_plot,
        c=param_plot,
        cmap='Greens',
        norm=norm,
        s=28,
        edgecolors='none',
        zorder=3
    )

    # --------------------------------------------------------
    # Mark start point (smallest parameter)
    # --------------------------------------------------------
    ax.scatter(
        qd_plot[0],
        td_plot[0],
        s=80,
        marker='o',
        facecolors='white',
        edgecolors='black',
        linewidths=1.3,
        zorder=5
    )

    # --------------------------------------------------------
    # Mark end point (largest parameter)
    # --------------------------------------------------------
    ax.scatter(
        qd_plot[-1],
        td_plot[-1],
        s=90,
        marker='s',
        facecolors='#2E8B57',
        edgecolors='black',
        linewidths=1.0,
        zorder=5
    )

    # --------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, zorder=0)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, zorder=0)

    # Set limits automatically first
    ax.relim()
    ax.autoscale_view()

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    diag_min = max(xmin, ymin)
    diag_max = min(xmax, ymax)

    if diag_min < diag_max:
        ax.plot(
            [diag_min, diag_max],
            [diag_min, diag_max],
            linestyle='--',
            linewidth=0.9,
            color='gray',
            alpha=0.7,
            zorder=0
        )

    # --------------------------------------------------------
    # Labels and title
    # --------------------------------------------------------
    ax.set_xlabel('QD', fontsize=12)
    ax.set_ylabel('TD', fontsize=12)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    if problem == 'single':
        source_text = 'Single source'
    else:
        source_text = 'Multiple sources'

    if noise_type == 'PP':
        noise_text = 'Probabilistic propagation'
    else:
        noise_text = 'Observation error'

    ax.set_title(f'({panel_label}) {source_text} under {noise_text}', fontsize=12)

    # Parameter range text
    ax.text(
        0.03, 0.96,
        parameter_range_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10
    )

    # Light grid
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.35)

    # --------------------------------------------------------
    # Add small colorbar for the panel
    # --------------------------------------------------------
    cbar = plt.colorbar(
        ScalarMappable(norm=norm, cmap='Greens'),
        ax=ax,
        fraction=0.046,
        pad=0.04
    )
    cbar.set_label(parameter_name, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    return qd_plot, td_plot, param_plot


# ============================================================
# 6. Create the 4-panel figure
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
axes = axes.flatten()

panel_settings = [
    ('single', 'PP', 'a'),
    ('single', 'OE', 'b'),
    ('multiple', 'PP', 'c'),
    ('multiple', 'OE', 'd')
]

export_rows = []

for ax, (problem, noise_type, panel_label) in zip(axes, panel_settings):
    qd_plot, td_plot, param_plot = plot_maps_trajectory(ax, problem, noise_type, panel_label)

    for qd_val, td_val, param_val in zip(qd_plot, td_plot, param_plot):
        export_rows.append({
            'Problem': problem,
            'Setting': noise_type,
            'Method': 'MAPS',
            'Parameter': param_val,
            'QD': qd_val,
            'TD': td_val
        })

# Global note
fig.suptitle(
    'QD–TD trajectories of MAPS under different propagation and observation settings',
    fontsize=14,
    y=0.98
)

# Add a figure-level note for start/end markers
fig.text(
    0.5, 0.015,
    'Open circle: smallest tested parameter value; filled square: largest tested parameter value.',
    ha='center',
    fontsize=10
)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# Save figure
plt.savefig('sensitivity_maps_qd_td_trajectory.png', dpi=600, bbox_inches='tight')
plt.show()


# ============================================================
# 7. Export QD-TD data
# ============================================================

trajectory_df = pd.DataFrame(export_rows)
trajectory_df.to_excel('results/maps_qd_td_sensitivity_data.xlsx', index=False)

print('\nSaved figure to: sensitivity_maps_qd_td_trajectory.png')
print('Saved trajectory data to: results/maps_qd_td_sensitivity_data.xlsx')