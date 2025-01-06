# NWM Catchment Network LSTM: Predicting Upstream Flows

## Description
The purpose of this project is to advance the work done by [Ramirez Molina et al. (2024)](https://github.com/aarm1978/Synthetic_Stream_Gauges) and [Lee et al. (2024)](https://github.com/quinnylee/synthetic_stream_gages). These projects developed LSTM models to predict ungaged upstream flow values when given a downstream flow value. Data was generated using [Frame et al. (2023)'s](https://github.com/NWC-CUAHSI-Summer-Institute/deep_bucket_lab) Deep Bucket Lab, which generated synthetic hydrographs using a "leaky bucket" model. The original code that the rest of the repository takes inspiration from is aarm_network_model.ipynb.

Initial testing using synthetic data showed promising results. This project implements Phase 2 of testing by using data from the [National Water Model retrospective data set](https://github.com/NOAA-Big-Data-Program/bdp-data-docs/blob/main/nwm/README.md) instead of data generated from the bucket models. Phase 3 of testing will be addressed in a later project and will use real-world gaged data.

This model is still actively under development.

## data_collection_preprocess
This folder contains files used to collect and preprocess forcing and attribute data from the National Water Model (NWM).

### Shapefiles
Shapefiles are used to generate random coordinates for study site selection. Shapefiles were sourced from the U.S. Census Bureau.

tl_2016_01_cousub: Shapefile for the state of Alabama
tl_2023_us_state: Shapefile of the United States

### all_data.csv
This file contains almost 5,000 randomly selected pairs of stream reaches and their corresponding NWM catchments across the continental United States. Under the column "du", "d" indicates that it is a downstream reach, and "u" indicates that it is an upstream reach. Each pair is listed as a consecutive "du" pair. For example, at the top of the csv, cat-2527958 & cat-2528001 form a pair, and the next two reaches cat-2873598 & cat-2873590 form a pair. The coordinate for each point is also listed, as well as the NHD COMID. This file can be used for experimenting with preprocessing or various scripts.

### coords_forcings_atts.sh
This shell script contains a command to use the ngiab_data_preprocess package to find all NWM catchments upstream of a given coordinate. The script requires the input of a path to a csv with coordinates. 

### Using the data preprocessing notebook

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

## data_examples

Contains several (very small) datasets that can be used to test out the model.

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

## Questions?

Contact Quinn Lee (data collection and preprocessing) at qylee@crimson.ua.edu or Sonam Lama (machine learning modeling) at slama@crimson.ua.edu.