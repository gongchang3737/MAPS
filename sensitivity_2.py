import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ============================================================
# 1. Basic settings
# ============================================================

LBL = ['SL', 'MLSL']
problem_list = ['single', 'multiple']
noise_list = ['PP', 'OE']
method = 'ILP'   # only ILP vs MLSL-ILP (MAPS)

# Parameter grids stored in the JSON files
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
# 2. Read results from JSON files
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
# 3. Compute QD and TD for MAPS
# ============================================================

def calculate_qd_td(problem, noise_type):
    """
    QD = (Q_MS - Q_SS) / Q_SS
    TD = (T_MS - T_SS) / T_SS

    Here:
    - SS = single-scale ILP
    - MS = multiscale ILP (MAPS)
    """
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
# 4. Parameter range used in the manuscript
# ============================================================

def get_parameter_range(noise_type):
    """
    Use the same parameter windows as in the manuscript:
    - PP: P_r in [0.70, 1.00]
    - OE: P_e in [0.00, 0.30]
    """
    if noise_type == 'PP':
        indices = np.arange(20, 51)   # 0.70 to 1.00
        x = quality_data.loc[indices, 'prop_prob_list'].to_numpy()
        x_label = r'$P_r$'
        x_range_text = r'$P_r: 0.70 \rightarrow 1.00$'
    else:
        indices = np.arange(0, 31)    # 0.00 to 0.30
        x = quality_data.loc[indices, 'error_prop_list'].to_numpy()
        x_label = r'$P_e$'
        x_range_text = r'$P_e: 0.00 \rightarrow 0.30$'

    return indices, x, x_label, x_range_text


# ============================================================
# 5. Plot one panel: MAPS QD and TD vs parameter
# ============================================================

def plot_qd_td_panel(ax, problem, noise_type, panel_label):
    qd, td = calculate_qd_td(problem, noise_type)
    indices, x, x_label, x_range_text = get_parameter_range(noise_type)

    qd_plot = qd.loc[indices].to_numpy(dtype=float)
    td_plot = td.loc[indices].to_numpy(dtype=float)
    x_plot = x.copy()

    valid = np.isfinite(qd_plot) & np.isfinite(td_plot)
    qd_plot = qd_plot[valid]
    td_plot = td_plot[valid]
    x_plot = x_plot[valid]

    # QD curve
    line_qd, = ax.plot(
        x_plot,
        qd_plot,
        marker='o',
        markersize=4,
        linewidth=1.8,
        label='QD'
    )

    # TD curve
    line_td, = ax.plot(
        x_plot,
        td_plot,
        marker='s',
        markersize=4,
        linewidth=1.8,
        linestyle='--',
        label='TD'
    )

    # Reference line y = 0
    ax.axhline(
        y=0,
        color='gray',
        linestyle='-',
        linewidth=0.8
    )

    # Labels
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('QD / TD', fontsize=11)

    # Percent formatting
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    # Fixed y range: -105% to 0.05%
    ax.set_ylim(-1.05, 0.05)

    # Title
    if problem == 'single':
        source_text = 'Single source'
    else:
        source_text = 'Multiple sources'

    if noise_type == 'PP':
        noise_text = 'Probabilistic propagation'
    else:
        noise_text = 'Observation error'

    ax.set_title(
        f'({panel_label}) {source_text} & {noise_text}',
        fontsize=12
    )

    # Parameter range text
    # ax.text(
    #     0.03,
    #     0.96,
    #     x_range_text,
    #     transform=ax.transAxes,
    #     ha='left',
    #     va='top',
    #     fontsize=10
    # )

    # Grid
    ax.grid(
        True,
        linestyle=':',
        linewidth=0.5,
        alpha=0.35
    )

    return line_qd, line_td, x_plot, qd_plot, td_plot


# ============================================================
# 6. Create the 4-panel figure
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
axes = axes.flatten()

panel_settings = [
    ('single', 'PP', 'a'),
    ('single', 'OE', 'b'),
    ('multiple', 'PP', 'c'),
    ('multiple', 'OE', 'd')
]

legend_handles = None
export_rows = []

for ax, (problem, noise_type, panel_label) in zip(axes, panel_settings):
    line_qd, line_td, x_plot, qd_plot, td_plot = plot_qd_td_panel(
        ax, problem, noise_type, panel_label
    )

    if legend_handles is None:
        legend_handles = [line_qd, line_td]

    for x_val, qd_val, td_val in zip(x_plot, qd_plot, td_plot):
        export_rows.append({
            'Problem': problem,
            'Setting': noise_type,
            'Method': 'MAPS',
            'Parameter': x_val,
            'QD': qd_val,
            'TD': td_val
        })


# Figure-level legend
fig.legend(
    legend_handles,
    ['QD', 'TD'],
    loc='upper center',
    bbox_to_anchor=(0.5, 0.985),
    ncol=2,
    frameon=False,
    fontsize=11
)

# Figure-level title
# fig.suptitle(
#     'Sensitivity of MAPS measured by QD and TD',
#     fontsize=14,
#     y=0.995
# )

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
plt.savefig(
    'sensitivity_maps_qd_td_curves.pdf',
    dpi=600,
    bbox_inches='tight'
)

plt.show()


# ============================================================
# 7. Export QD and TD data
# ============================================================

export_df = pd.DataFrame(export_rows)
export_df.to_excel(
    'results/maps_qd_td_curves_data.xlsx',
    index=False
)

print('\nSaved figure to: sensitivity_maps_qd_td_curves.pdf')
print('Saved data to: results/maps_qd_td_curves_data.xlsx')