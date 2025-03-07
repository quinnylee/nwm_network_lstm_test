import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import shutil
import time
import datetime
# load libraries for pair generation
from concurrent.futures import ThreadPoolExecutor
import random
import multiprocessing as mp
import xarray as xr
import os
import sys
import glob
from exactextract import exact_extract
import warnings
import json
from tqdm import tqdm  # Add tqdm


def read_key_value_file(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    
    key_value_list = [(key, value) for key, value in data.items()]
    return key_value_list

def process_chrtout_file2(filename, id_list, value_col="streamflow"):
    with xr.open_dataset(filename) as ds:
        subset = ds[[value_col]].sel(feature_id=id_list)
        return subset

def get_q(qlat_files, id_list, value_col="streamflow"):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(
            process_chrtout_file2,
            [(filename, id_list, value_col) for filename in qlat_files]
        )
    combined = xr.concat(results, dim="time")
    return combined

start = time.perf_counter()

file_path = "/media/volume/NeuralHydrology/Test_Quinn_Data/all_pairs.txt"
key_value_pairs = read_key_value_file(file_path)

comids = []

for i, (key, value) in enumerate(key_value_pairs):
    comids.append(key)
    comids.append(value)

comids = [3441628, 3438398]
q_dataset = get_q(glob.glob("/mnt/IMP_DATA/CHRTOUT/2008/*.CHRTOUT_DOMAIN1"), comids)
target_directory = '/media/volume/NeuralHydrology/Test_Quinn_Data/Channel_Routing/2008/'
feature_id_list = q_dataset['feature_id'].values

# Add tqdm progress bar here
for i in tqdm(range(len(feature_id_list)), desc="Saving NetCDF files"):
    file_path = target_directory + str(feature_id_list[i]) + '.nc'
    q_dataset.isel(feature_id=i).to_netcdf(file_path)

end = time.perf_counter()

print(f"Operation took {end - start:.2f} seconds")
