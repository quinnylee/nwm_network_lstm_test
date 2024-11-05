# NWM Catchment Network LSTM: Predicting Upstream Flows

## Description
The purpose of this project is to advance the work done by [Ramirez Molina et al. (2024)](https://github.com/aarm1978/Synthetic_Stream_Gauges) and [Lee et al. (2024)](https://github.com/quinnylee/synthetic_stream_gages). These projects developed LSTM models to predict ungaged upstream flow values when given a downstream flow value. Data was generated using [Frame et al. (2023)'s](https://github.com/NWC-CUAHSI-Summer-Institute/deep_bucket_lab) Deep Bucket Lab, which generated synthetic hydrographs using a "leaky bucket" model.

Initial testing using synthetic data showed promising results. This project implements Phase 2 of testing by using data from the [National Water Model retrospective data set](https://github.com/NOAA-Big-Data-Program/bdp-data-docs/blob/main/nwm/README.md) instead of data generated from the bucket models. Phase 3 of testing will be addressed in a later project and will use real-world gaged data.

This model is still actively under development.

## LSTM model notes

You can simply clone this repository to access the LSTM, which is in the notebook, and several small preliminary data sets. 
The following packages are required:
- numpy
- pandas
- matplotlib
- IPython
- math
- torch
- scikit-learn
- tqdm

Additionally, the LSTM can run on a CPU, but is optimized for a machine with a CUDA-enabled GPU or Apple MPS.

You can simply change out the file path to the data at the beginning of section 1.2. There may be errors with names of model parameters, but those are easy to fix.

## Using the data preprocessing notebook

This notebook assumes that you have a good amount of NetCDF files downloaded from the NOAA NWM retrospective AWS bucket and that you want to preprocess these files into a format that the LSTM model code likes. It also assumes that you want a random sample of points from some shapefile. You will need a shapefile of some area within CONUS. You will need the following packages:
- pandas
- geopandas
- numpy
- shapely
- shutil
- datetime
- random
- multiprocessing
- xarray
- os
- sys
- glob
- ngiab_data_preprocess

The notebook will generate random points from a shapefile, find its corresponding catchment (arbitrarily considered downstream), select an upstream catchment close to each downstream catchment, and collect forcing/streamflow/attribute data from the NWM NetCDF files into a nice CSV.

Note: the notebook requires some extra work on the user's end. An intermediate step requires the use of an R package or GIS software (R recommended).
