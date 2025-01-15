'''
Given a csv file of study site pairs and their COMIDs, this script can be used 
to preprocess NWM forcings and attributes for the purposes of an ML model. 
Please make sure you have the Lynker NWM hydrofabric, FORCING and CHRTOUT files 
from the NWM retrospective dataset, and catchment attributes downloaded, and 
make sure you change the filepaths in the appropriate lines.

This script is not functional yet!!!

To run the script, use this command:
python data_process.py [-h] [-r] time_filename

positional arguments:
  time_filename  Path to text file with list of time arguments 
    (ex. 2008/200801*)

options:
  -h, --help     show this help message and exit
  -r, --regular  Use non-parallelized method of saving data. Significantly 
    slower

01/13/2025
Quinn Lee - qylee@crimson.ua.edu
Sonam Lama - slama@crimson.ua.edu
'''

# Load libraries

import sys
import warnings
import setup
import forcs_q
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("time_filename", 
                    help="Path to text file with list of time arguments "\
                        "(ex. 2008/200801*)")
parser.add_argument("-r", "--regular", action="store_true",
                    help="Use non-parallelized method of saving data. " \
                        "Significantly slower")
args = parser.parse_args()

# Sets string for (globbed) time period selection for FORCING and CHRTOUT files 
# from system args
time_filename = args.time_filename
with open(time_filename, 'r') as time_file:
    time_args = time_file.read()
    time_args = time_args.split('\n')

# Ignores warnings about CRS mismatch. if we don't do this the output blows up
warnings.filterwarnings('ignore')

# Read catchment data
cats_path = 'all_data.csv'
catchments = setup.read_cats(cats_path)
comids = setup.get_comids(catchments)

# Read catchment attribute data
ngiab_output_dir = "/media/volume/Clone_Imp_Data/quinn_test_atts" \
    "/ngiab_preprocess_output/"
attr_paths, catids = setup.parallel_gca(catchments, ngiab_output_dir)

# Set up hydrofabric
hf_path = '/home/exouser/Downloads/conus_nextgen.gpkg'
hydrofabric = setup.hydrofabric_setup(hf_path)

# Set up data columns
forcing_vars = ['U2D', 'V2D', 'LWDOWN', 'RAINRATE', 'T2D', 'Q2D', 'PSFC', 
                'SWDOWN', 'LQFRAC']
colnames = setup.col_names(attr_paths, forcing_vars)

# Save data
for time_arg in time_args:
    # Open forcing files
    forc_dataset, times = forcs_q.forcing_process(time_arg)

    # Open CHRTOUT files
    q_dataset = forcs_q.chrtout_process(time_arg, comids)

    if args.regular:
        forcs_q.save_data_reg(time_arg, catchments, comids, attr_paths, times, 
                          colnames, hydrofabric, forcing_vars, forc_dataset, 
                          q_dataset)
    else:
        forcs_q.save_data_np(time_arg, catchments, comids, attr_paths, times, 
                          colnames, hydrofabric, forcing_vars, forc_dataset, 
                          q_dataset)