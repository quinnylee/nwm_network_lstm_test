'''
Given a csv file of study site pairs and their COMIDs, this script can be used to
preprocess NWM forcings and attributes for the purposes of an ML model. Please
make sure you have the Lynker NWM hydrofabric, FORCING and CHRTOUT files from the 
NWM retrospective dataset, and catchment attributes downloaded, and make sure you 
change the filepaths in the appropriate lines.

This script is not functional yet!!!

To run the script, use this command:
python 0108_data_process.py dates
example of a dates argument: 2008/200801* (pulls all time steps from January 2008)

01/09/2025
Quinn Lee - qylee@crimson.ua.edu
Sonam Lama - slama@crimson.ua.edu
'''

# Load libraries

import pandas as pd
import geopandas as gpd
import datetime
import multiprocessing as mp
import xarray as xr
import os
import sys
import glob
from exactextract import exact_extract
import warnings

# Sets string for (globbed) time period selection for FORCING and CHRTOUT files from system args
time_arg = sys.argv[1]

# Ignores warnings about CRS mismatch. if we don't do this the output blows up
warnings.filterwarnings('ignore')

# Reads information about desired catchments from csv
cats_path = 'path/to/csv'
catchments = pd.read_csv(cats_path)

# generate pair IDs for each upstream/downstream pair of catchments in case data gets shuffled around by accident
pairids = []
for i in range(len(catchments)):
    if i % 2 == 0:
        pairids.append(i / 2)
    else:
        pairids.append((i-1)/2)

catchments['pair_id'] = pairids

# create list of COMIDs
comids = catchments['comid'].tolist()

# path to catchment attributes
ngiab_output_dir = '/media/volume/Imp_Data/quinn_test_atts/ngiab_preprocess_output/'

# generate path to attribute files for one catchment, as well as CATID
def getclosest_array(i):
    catid = catchments.iat[i, 0]
    
    if catchments.iat[i,1] == 'd':
        attr_path = os.path.join(ngiab_output_dir, catid, 'config', 'cfe_noahowp_attributes.csv') 
    else:
        ds_catid = catchments.iat[i-1,0]
        attr_path = os.path.join(ngiab_output_dir, ds_catid, 'config', 'cfe_noahowp_attributes.csv')
    
    attr_df = pd.read_csv(attr_path)
    attr_df.set_index('divide_id', inplace=True)

    return attr_path, catid

# generates attribute filepaths and CATIDs for all catchments in parallel
def parallel_gca():
    with mp.Pool() as pool:
        result = pool.map(getclosest_array, range(len(catchments)))
    attr_paths, catids = zip(*result)
    return attr_paths, catids

attr_paths, catids = parallel_gca()

# use globbed filepath, opens forcing NC files 
forc_path = "/media/volume/Imp_Data/FORCING/" + time_arg + ".LDASIN_DOMAIN1"
forc_dataset = xr.open_mfdataset(forc_path)

# get timestamps for all time steps in study period
times = forc_dataset.time.values

# get time strings
time_strs = []
for i in range(len(times)):
    pddt = pd.to_datetime(times[i])
    newstr = pddt.strftime("%Y-%m-%d %H:%M:%S")
    time_strs.append(newstr)

# process CHANNEL ROUTING (CHRTOUT) files, returns df of streamflow values
problem_comids = []

# gets all relevant streamflow data from one timestep (one CHRTOUT file)
def process_chrtout_file(filename, id_list, value_col="streamflow"):
    with xr.open_dataset(filename) as ds:
        df_list = []
        try:
            for id in id_list:
                df_cat = ds[["time", value_col]].sel(feature_id=id).to_dataframe()
                df_list.append(df_cat)
        except:
            problem_comids.append(id) 
            print(id, " not found")
            #id_list.remove[id]

        df = pd.concat(df_list)
        return df

# gets all relevant streamflow data from multiple timesteps through parallelization
def get_q(qlat_files, id_list, index_col="feature_id", value_col="streamflow"):

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(process_chrtout_file, [(filename, id_list) for filename in qlat_files])

    frame = pd.concat(results, axis=0, ignore_index=False)

    return frame

# use globbed filepath to generate q_dataset
chrtout_path = "/media/volume/Imp_Data/CHRTOUT/" + time_arg + ".CHRTOUT_DOMAIN1"
q_dataset = get_q(glob.glob(chrtout_path), comids)
q_dataset.reset_index(inplace=True)

# define column names for final dataset
forcing_vars = ['U2D', 'V2D', 'LWDOWN', 'RAINRATE', 'T2D', 'Q2D', 'PSFC', 'SWDOWN', 'LQFRAC']
colnames = []
ex_attr_df = pd.read_csv(attr_paths[0])
ex_attr_df.set_index('divide_id', inplace=True)
colnames.append("comid")
colnames.append("catid")
colnames.append("du")
colnames.append("pair_id")
for var_name in forcing_vars:
    colnames.append(var_name)
colnames.append("streamflow")
for var_name in ex_attr_df.columns:
    colnames.append(var_name)

# Set up hydrofabric
# change path to your hydrofabric download location
hf_path = '/home/exouser/Downloads/conus_nextgen.gpkg'
hydrofabric = gpd.read_file(hf_path)

# reproject to match forcing dataset CRS
hydrofabric = hydrofabric.to_crs("+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext  +no_defs")

# Collect forcings, attributes, and streamflow data for a catchment at one time step
def process_time(t_str, t, lat, lon, comid, du, pair_id, catid, attr_values):

    # Initialize the row with basic catchment information
    row = [comid, catid, du, pair_id]

    forcing_data = []
    gdf = hydrofabric.loc[hydrofabric['divide_id']==catid]
    for var in forcing_vars:
        raster = forc_dataset.sel(time=t)[var]
        raster = raster.rio.write_crs("+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext  +no_defs")
        zonal_stats = exact_extract(raster, gdf, 'mean', include_cols='divide_id', output='pandas')
        forcing_data.append(zonal_stats['mean'][0])
    # Append forcing data for each variable
    for var_value in forcing_data:
        row.append(var_value)

    # Streamflow data for the given time and comid
    filtered_data = q_dataset[(q_dataset['time']==t) & (q_dataset['feature_id']==comid)]
    streamflow_value = filtered_data['streamflow'].iloc[-1]
    row.append(streamflow_value)
    #print(t)

    # Append attributes (assuming these were preloaded)
    row.extend(attr_values)

    return (t, row)

# Collects forcings, attributes, streamflow data for one catchment over all time steps 
def process_catchment(k):
    df = {}
    lat = catchments['y'][k]
    lon = catchments['x'][k]
    comid = comids[k]
    attr_path = attr_paths[k]
    catid = catchments['catchment_id'][k]
    du = catchments['du'][k]
    pair_id = catchments['pair_id'][k]
    attr_df = pd.read_csv(attr_path)
    attr_df.set_index('divide_id', inplace=True)
    attr_values = attr_df.loc[catid].tolist()

    #print("loaded atts")

    args = [(time_strs[i], times[i], lat, lon, comid, du, pair_id, catid, attr_values)
            for i in range(len(time_strs))]
    
    with mp.Pool(processes=30) as pool:
        results = pool.starmap(process_time, args)

    #print("finished mp")
    df = {t: row for t, row in results}
    df = pd.DataFrame.from_dict(df, orient='index', columns=colnames)
    return df

# Collects and saves data for all catchments over all timesteps
# Outputs a file that logs errors and progress
output_name = 'output' + datetime.date.today().isoformat() + '.txt'
results = []
for i in range(len(catchments)):
    try:
        result = process_catchment(i)
        results.append(result)
        with open(output_name, 'a') as file:
            toprint = str(i) + ' done\n'
            file.write(toprint)
    except:
        with open(output_name, 'a') as file:
            toprint = str(i) + ' had an error\n'
            file.write(toprint)
        continue

try:
    data = pd.concat(results)
    with open(output_name, 'a') as file:
        file.write('df concatenated\n')
except Exception as e:
    with open(output_name, 'a') as file:
        file.write(f"Error: {e}\n")

try:
    data.to_csv('runs/experiment_2024-10-15/jan_data_agg.csv')
    with open(output_name, 'a') as file:
        file.write('saved :D\n')
except Exception as e:
    with open(output_name, 'a') as file:
        file.write(f"Error: {e}\n")