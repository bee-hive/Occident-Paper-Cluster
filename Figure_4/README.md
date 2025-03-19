## Figure 4

#### Order for running python scripts

1. `t_cell_segmentation_morphology.py`
    - Inputs:
        - `directory_path` (line 407): path do the directory that contains the segmentation results for all wells. Zip files in this directory correspond to 50-frame intervals of each well. Standard naming convention is `cart_B10_start_0_end_50_nuc_15_cyto_75.zip` which corresponds to well `B10` between time frames `0` and `50`.
    - Outputs:
        - Plots (pdf files) and tables (csv files) specified from lines 62-74. Note that the location of the `.plots/` and `.tables/` directories will be relative to wherever you execute this python script.
    - Execution:
        - `python3 t_cell_segmentation_morphology.py`
    - Notes:
        - This step is listed first in the Fig 4 pipeline but its inputs and outputs are not required in any other Fig 4 python scripts or notebooks. Therfore, the order that this step is run is not important.
2. `cell_tracking_velocity_data.py`
    - Inputs:
    - Outputs:
    - Execution:
        - `python3 cell_tracking_velocity_data.py`
    - Notes:
        - This step takes a long time to run (approx 48 hours on a beehive server CPU node). It is therefore recommended to run this script in a tmux window.
3. `cell_tracking_interactions.py`
    - Inputs:
    - Outputs:
    - Execution:
        - `python3 cell_tracking_interactions.py`
    - Notes:
4. `interactions.ipynb`
