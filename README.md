# MAPS
MAPS: Multiscale Source Detection in Multilayer Affiliation Networks

Last Updated: Sept 19, 2026

# Framework Structure Overview

The process framework is hierarchically structured into 3 levels of granularity. Each network node is identified by a 6-digit ID, where every 2 digits represent a specific hierarchical level. Currently, the first 4 digits are actively used, defining the first and second-level nodes. The 5th and 6th digits are temporarily reserved (left blank) to accommodate finer-grained node definitions in future extensions.

# Data Description

The main input data is stored in the `data` directory:

**`nodelist_L1-3.xlsx`**: Contains the node IDs (`id`) for the first, second, and third-level nodes.

**`edgelist_L1-3.xlsx`**: Contains the network edge information across the three levels, including the source node (`source`) and the target node (`target`).

# Code Description

**`SL/MLSL_single/multiple_anynoise/OEPP.py`**: The main execution scripts for source detection tasks under various problem settings and noise scenarios.

**`plot_res.py`**: Used to plot and visualize the experimental output results.

**`gene_large_graph.py`**: The script used to generate the large-scale graph ($G_3$) from the basic graph ($G_2$).

**`sensitivity.py`**: The script for conducting sensitivity analysis experiments.

**`guro_opt`**: The optimization code utilizing the Gurobi solver.

# Results Description

The main output data is saved in the `results` directory. This includes the source detection task results, solution quality metrics, and computational time data under different problem settings and scenarios.
