from s3fs import S3FileSystem
from s3fs.core import _error_wrapper, version_id_kw
from typing import Optional
import asyncio
import sqlite3
import os
import s3fs
import xarray as xr
import logging
from dask.distributed import Client, LocalCluster, progress
from distributed.utils import silence_logging_cmgr
from pathlib import Path
import pandas as pd
import glob
import time
import json
import numpy as np
import warnings
from hydrotools.nwis_client import IVDataService
import hydroeval as he
from colorama import Fore, Style, init
import multiprocessing
from functools import partial

from s3fs import S3FileSystem
from s3fs.core import _error_wrapper, version_id_kw
from typing import Optional
import asyncio



class S3ParallelFileSystem(S3FileSystem):
    """S3FileSystem subclass that supports parallel downloads"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _cat_file(
        self,
        path: str,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> bytes:
        bucket, key, vers = self.split_path(path)
        version_kw = version_id_kw(version_id or vers)

        # If start/end specified, use single range request
        if start is not None or end is not None:
            head = {"Range": await self._process_limits(path, start, end)}
            return await self._download_chunk(bucket, key, head, version_kw)

        # For large files, use parallel downloads
        try:
            obj_size = (
                await self._call_s3(
                    "head_object", Bucket=bucket, Key=key, **version_kw, **self.req_kw
                )
            )["ContentLength"]
        except Exception as e:
            # Fall back to single request if HEAD fails
            return await self._download_chunk(bucket, key, {}, version_kw)

        CHUNK_SIZE = 5 * 1024 * 1024  # 1MB chunks
        if obj_size <= CHUNK_SIZE:
            return await self._download_chunk(bucket, key, {}, version_kw)

        # Calculate chunks for parallel download
        chunks = []
        for start in range(0, obj_size, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE - 1, obj_size - 1)
            range_header = f"bytes={start}-{end}"
            chunks.append({"Range": range_header})

        # Download chunks in parallel
        async def download_all_chunks():
            tasks = [
                self._download_chunk(bucket, key, chunk_head, version_kw) for chunk_head in chunks
            ]
            chunks_data = await asyncio.gather(*tasks)
            return b"".join(chunks_data)

        return await _error_wrapper(download_all_chunks, retries=self.retries)

    async def _download_chunk(self, bucket: str, key: str, head: dict, version_kw: dict) -> bytes:
        """Helper function to download a single chunk"""

        async def _call_and_read():
            resp = await self._call_s3(
                "get_object",
                Bucket=bucket,
                Key=key,
                **version_kw,
                **head,
                **self.req_kw,
            )
            try:
                return await resp["Body"].read()
            finally:
                resp["Body"].close()

        return await _error_wrapper(_call_and_read, retries=self.retries)


def save_to_cache(stores: xr.Dataset, cached_nc_path: Path) -> xr.Dataset:
    """Compute the store and save it to a cached netCDF file. This is not required but will save time and bandwidth."""
    logging.info("Downloading and caching forcing data, this may take a while")

    if not cached_nc_path.parent.exists():
        cached_nc_path.parent.mkdir(parents=True)

    # sort of terrible work around for half downloaded files
    temp_path = cached_nc_path.with_suffix(".downloading.nc")
    if os.path.exists(temp_path):
        os.remove(temp_path)

    ## Cast every single variable to float32 to save space to save a lot of memory issues later
    ## easier to do it now in this slow download step than later in the steps without dask
    for var in stores.data_vars: 
        stores[var] = stores[var].astype("float32")

    try:
        client = Client.current()
    except ValueError:
        cluster = LocalCluster()
        client = Client(cluster)

    future = client.compute(stores.to_zarr(temp_path, compute=False))
    # Display progress bar
    progress(future)
    future.result()

    os.rename(temp_path, cached_nc_path)

    # data = xr.open_mfdataset(cached_nc_path, parallel=True, engine="h5netcdf")
    # return data


def read_key_value_file(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    
    key_value_list = [(key, value) for key, value in data.items()]
    return key_value_list

def download_nwm_output(start_time, end_time, feature_ids) -> xr.Dataset:
    """Load zarr datasets from S3 within the specified time range."""
    # if a LocalCluster is not already running, start one
    try:
        client = Client.current()
    except ValueError:
        cluster = LocalCluster()
        client = Client(cluster)

    logging.debug("Creating s3fs object")
    store = s3fs.S3Map(
        f"s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
        s3=S3ParallelFileSystem(anon=True),
        # s3=S3FileSystem(anon=True),
    )

    logging.debug("Opening zarr store")
    dataset = xr.open_zarr(store, consolidated=True)

    # select the feature_id
    logging.debug("Selecting feature_id")
    dataset = dataset.sel(time=slice(start_time, end_time), feature_id=feature_ids)

    # drop everything except coordinates feature_id, gage_id, time and variables streamflow
    dataset = dataset[["streamflow"]]
    logging.debug("Computing dataset")
    logging.debug("Dataset: %s", dataset)

    return dataset

def main():
    warnings.filterwarnings("ignore", message="No data was returned by the request.")
    
    # Initialize colorama
    init(autoreset=True)

    logging.basicConfig(level=logging.INFO)

    file_path = "/media/volume/NeuralHydrology/Test_Quinn_Data/all_pairs.txt"
    key_value_pairs = read_key_value_file(file_path)
    comids = []
    for (key,value) in key_value_pairs:
        comids.append(int(key))
        comids.append(value)

    dataset = download_nwm_output(start_time= '1979-10-01', end_time= '2024-09-30', feature_ids=comids)
    dataset = dataset.drop_vars(['elevation', 'latitude', 'longitude', 'order', 'gage_id'])

    dataset = dataset.chunk({'time': 100, 'feature_id': 1000}) 
    save_to_cache(dataset, Path("/mnt/IMP_DATA/CHROUT.zarr"))

if __name__ == "__main__":
    main()
