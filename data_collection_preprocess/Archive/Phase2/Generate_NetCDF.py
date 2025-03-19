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
import logging
import dask.array as da
import zarr
from dask.distributed import LocalCluster, Client
import dask

def read_key_value_file(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    
    key_value_list = [(key, value) for key, value in data.items()]
    return key_value_list

def process_chrtout_file2(filename, id_list, value_col="streamflow"):
    # with xr.open_dataset(filename) as ds:
    #     subset = ds[[value_col]].sel(feature_id=id_list)
    #     subset = subset.to_dataarray()
    #     return subset

    """Reads and subsets CHRTOUT files using Dask for lazy loading."""
    ds = xr.open_dataset(filename, engine="h5netcdf", chunks={})  # Lazy loading
    return ds[value_col].sel(feature_id=id_list)

def get_q(qlat_files, id_list, value_col="streamflow"):
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     results = pool.starmap(
    #         process_chrtout_file2,
    #         tqdm(
    #             ((filename, id_list, value_col) for filename in qlat_files), 
    #             total=len(qlat_files),
    #             desc="Opening and subsetting CHRTOUT files")
    #     )

    # combined = dask.array.concatenate(results)
    # return combined

    """Parallelized function to read CHRTOUT files lazily using Dask."""
    delayed_loads = [dask.delayed(process_chrtout_file2)(f, id_list, value_col) for f in qlat_files]
     # Convert each delayed object into a 2D Dask array (num_files, num_comids)
    dask_arrays = [
        da.from_delayed(d, shape=(len(id_list),), dtype=np.float32) for d in delayed_loads
    ]

    # Stack along the time dimension instead of concatenating
    combined = da.stack(dask_arrays, axis=0) #.rechunk((100, len(id_list)))  # Shape: (num_files, num_comids)
    return combined

def save_zarr(q_dataset, comids, target_directory): # new function I added to use dask

    save_tasks = []
    q_dataset = q_dataset.rechunk({0: "auto", 1: 10})
    q_datset = q_dataset.persist()

    # for i in tqdm(range(q_dataset.shape[1]), desc="Saving zarr files"):
    #     file_path = os.path.join(target_directory, f"{comids[i]}.zarr")
        
    #             # Rechunk and delay execution
    #     task = dask.delayed(q_dataset[:, i].rechunk((1000,)).to_zarr)(file_path, compute=False)
    #     save_tasks.append(task)
    batch_size = 10  # Adjust based on memory
    for start in tqdm(range(0, len(comids), batch_size), desc="Saving Zarr files"):
        batch_comids = comids[start:start + batch_size]
        file_path = os.path.join(target_directory, f"batch_{start}.zarr")
        
        # Select batch, rechunk, and write
        batch_data = q_dataset[:, start:start + batch_size]
        batch_data = batch_data.rechunk({0: "auto"})
        task = dask.delayed(batch_data.to_zarr)(file_path, compute=False)
        save_tasks.append(task)
        # batch_data.to_zarr(file_path)

    dask.compute(*save_tasks)  # Process all writes in parallel efficiently

def main():
    logging.basicConfig(level=logging.DEBUG)
    start = time.perf_counter()

    file_path = "/media/volume/NeuralHydrology/Test_Quinn_Data/all_pairs.txt"
    logging.debug(f"file path: {file_path}")
    key_value_pairs = read_key_value_file(file_path)

    comids = []

    for i, (key, value) in enumerate(key_value_pairs):
        comids.append(int(key))
        comids.append(value)

    #comids = [3441628, 3438398]
    # logging.debug(f'got ids')

    # q_dataset = get_q(glob.glob("/mnt/IMP_DATA/CHRTOUT/2008/*.CHRTOUT_DOMAIN1"), comids)
    # logging.info("finished getting one year of streamflow")

    # logging.debug(type(q_dataset))
    # logging.debug(q_dataset)
    # target_directory = '/media/volume/NeuralHydrology/Test_Quinn_Data/Channel_Routing/2008/'

    # logging.info("starting process")
    # # Add tqdm progress bar here
    # with LocalCluster() as cluster:
    #     with Client(cluster) as client:
    #         for i in tqdm(range(len(comids)), desc="Saving zarr files"):
    #             file_path = target_directory + str(comids[i]) + '.zarr'
    #             q_dataset_basin = q_dataset[:, i]
    #             q_dataset_basin.to_zarr(file_path)

    with LocalCluster(n_workers=24, memory_limit="4GB") as cluster, Client(cluster) as client:
        logging.debug("clusters started")

        qlat_files = glob.glob("/mnt/IMP_DATA/CHRTOUT/2009/*.CHRTOUT_DOMAIN1")
        q_dataset = get_q(qlat_files, comids)
        logging.debug(f"q_dataset shape: {q_dataset.shape}")
        logging.info("finished getting one year of streamflow (get_q)")

        logging.debug(type(q_dataset))
        logging.debug(q_dataset)

        target_directory = '/media/volume/NeuralHydrology/Test_Quinn_Data/Channel_Routing/2009/'
        os.makedirs(target_directory, exist_ok=True)

        logging.info("starting zarr save process")

        save_zarr(q_dataset, comids, target_directory)


    end = time.perf_counter()

    print(f"Operation took {end - start:.2f} seconds")

if __name__ == "__main__":
    main()