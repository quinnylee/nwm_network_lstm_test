# NWM Catchment Network LSTM: Predicting Upstream Flows

## Description of repository
Our project's goal is to construct a dataset of historical gage estimates for ungaged basins across the NextGen domain using machine learning (ML) techniques to assimilate known gage values from nearby basins into our predictions. ML models have been shown to predict flows well in ungaged basins (Kratzert et al., 2019; see also Ghaneei & Moradkhani, 2025; Frame et al., 2025), and if we constrain the ML training with information from observed flows in the same network, we can increase the accuracy of the ungaged flow estimates (Fisher et al., 2020). 

We have conducted experiments with synthetic hydrology to show that, with two-basin networks, one downstream and one upstream, an ML model can be trained to provide a higher accuracy estimate of the upstream flows from "ungaged" synthetic basins than can be obtained by training an ML model on individual basins alone [(Ramirez Molina et al., 2024; ](https://github.com/aarm1978/Synthetic_Stream_Gauges) [Lee et al., 2024)](https://github.com/quinnylee/synthetic_stream_gages). Building on these synthetic experiments, we propose to expand this proof-of-concept to include more robust training using National Water Model (NWM) data.

This project uses data from the [NWM 3.0 retrospective dataset](https://github.com/NOAA-Big-Data-Program/bdp-data-docs/blob/main/nwm/README.md) instead of synthetic data. We used the NWM 3.0 channel hydrofabric to select adjacent basin pairs (one upstream and one downstream). We used PyTorch to create a long short-term memory (LSTM) model that takes meteorological forcings, basin attributes, and downstream streamflow values to predict our "ungaged" upstream value. We chose to work with the NWM 3.0 retrospective dataset because it gave us a near-infinite amount of data to train and validate our model, and because we had greater control over the study area and network geometries. This phase of the project will result in a prototype workflow and LSTM model for ML-assisted data assimilation. 

Phases 3 and 4 of this project will be addressed in later repositories and will use real-world gaged data. These phases will result in retrospective datasets of high-confidence estimates of flow at ungaged basins. These datasets can be used by the hydrologic community as robust evaluation datasets to tune predictive models for ungaged basins within the NextGen Framework. This workflow will also lay the foundation for future researchers interested in ML-assisted data assimilation across more complex network topologies.

This model is still actively under development.

## BIG_DATA
This directory contains bigger datasets to train the model. We will ideally want this to be in a volume outside of repo. But for now, since Quinn and I are working on it simultaneously and a volume cant be shared simultaneously, I have a year's data stored in here for now. 

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

## Results
This directory contains the preliminary results of combined model vs single model

## LSTM model notes

You can simply clone this repository to access the LSTM, which is in the 'data_train.ipynb' notebook, and several small preliminary data sets. 
The following packages are required:
- numpy
- pandas
- matplotlib
- IPython
- math
- torch
- scikit-learn

Additionally, the LSTM can run on a CPU, but is optimized for a machine with a CUDA-enabled GPU or Apple MPS.

You can simply change out the file path to the data at the beginning of section 1.2. There may be errors with names of model parameters, but those are easy to fix.

## my_script.sh
This is just a simple bash script yo transfer Forcing and Channel routing data from s3 bucket to volume.

## Questions?

Contact Quinn Lee (data collection and preprocessing) at qylee@crimson.ua.edu or Sonam Lama (machine learning modeling) at slama@crimson.ua.edu.