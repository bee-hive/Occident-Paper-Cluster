import numpy as np
from utils import load_data_local
import time
import argparse

parser = argparse.ArgumentParser(
                    prog='centroid calculation',
                    description='calculations and saves x,y centroid coordinates ',
                    epilog='Archit Verma 2024')
parser.add_argument('--well', default = 0, type = int, help = 'well ix')
args = parser.parse_args()

T0 = time.time()

wells = ['', 'B3', 'B4', 'B5', 'B6', 'E3', 'E4', 'E5', 'E6', 'B7', 'B8', 'B9', 'B10']
lower_lim = 0
upper_lim = 350

well = args.well
print(wells[well])
B3 = load_data_local('./updated_full/with_nuclei/cart_' + wells[well] + '_start_0_end_350_nuc_15_cyto_75.zip')

ts = B3['y'][lower_lim:upper_lim,:,:,0]
print(ts.shape)
cells = np.unique(ts)[1:]

centroids = np.empty((len(cells), upper_lim - lower_lim, 2))
centroids[:] = np.nan
for i in range(len(cells)):
    if i % 1000 == 0:
        print(i)
    cell_t, cell_x, cell_y = np.where(ts == cells[i])
    for t in np.unique(cell_t):
        mask = np.where(cell_t == t)
        x_mean = np.mean(cell_x[mask])
        y_mean = np.mean(cell_y[mask])
        centroids[i, t, 0] = x_mean
        centroids[i, t, 1] = y_mean


np.save('./centroid-arrays/' + wells[well] + '-nuclei', centroids)
print(time.time() - T0)