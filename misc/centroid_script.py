import numpy as np
from occident.utils import load_deepcell_object
from occident.centroids import centroid_calculation
import time
import argparse

parser = argparse.ArgumentParser(
                    prog='centroid calculation',
                    description='calculations and saves x,y centroid coordinates ',
                    epilog='Archit Verma 2024')
parser.add_argument('--well', default=0, type=int, help='integer index for the well_id')
parser.add_argument('--type', default='cancer', type=str, help='type of centroid array to calculate')
args = parser.parse_args()

T0 = time.time()

wells = ['', 'B3', 'B4', 'B5', 'B6', 'E3', 'E4', 'E5', 'E6', 'B7', 'B8', 'B9', 'B10']
assert args.well in range(1, 13), "Well integer index must be between 1 and 12."
well = args.well
print(wells[well])
# load the deepcell object for the specified well
dcl_ob = load_deepcell_object('./updated_full/cart_' + wells[well] + '_start_0_end_350_nuc_15_cyto_75.zip')

# make sure a valid type of centroid calculation is specified
assert args.type in ['cancer', 'T', 'nuclei'], "Type must be either 'cancer', 'T' or 'nuclei'."

# specify the time limits for the centroid calculation
lower_lim = 0
upper_lim = 350

# compute the centroids for the specified type using the occident centroid_calculation function
centroids = centroid_calculation(dcl_ob, type=args.type, lower_lim=lower_lim, upper_lim=upper_lim)

# save the centroids to a numpy file
if args.type == 'cancer':
    output_suffix = '-cancer-cells'
elif args.type == 'T':
    output_suffix = '-T-cells'
elif args.type == 'nuclei':
    output_suffix = '-nuclei'
np.save('./centroid-arrays/' + wells[well] + output_suffix, centroids)

print(time.time() - T0)