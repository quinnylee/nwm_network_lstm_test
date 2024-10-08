# NWM Catchment Network LSTM: Predicting Upstream Flows

## Description
The purpose of this project is to advance the work done by [Ramirez Molina et al. (2024)](https://github.com/aarm1978/Synthetic_Stream_Gauges) and [Lee et al. (2024)](https://github.com/quinnylee/synthetic_stream_gages). These projects developed LSTM models to predict ungaged upstream flow values when given a downstream flow value. Data was generated using [Frame et al. (2023)'s](https://github.com/NWC-CUAHSI-Summer-Institute/deep_bucket_lab) Deep Bucket Lab, which generated synthetic hydrographs using a "leaky bucket" model.

Initial testing using synthetic data showed promising results. This project implements Phase 2 of testing by using data from the [National Water Model retrospective data set](https://github.com/NOAA-Big-Data-Program/bdp-data-docs/blob/main/nwm/README.md) instead of data generated from the bucket models. Phase 3 of testing will be addressed in a later project and will use real-world gaged data.

This model is still actively under development.

## Requirements

You can simply clone this repository to access the LSTM, which is in the notebook, and several small preliminary data sets. 
The following packages are required:
-numpy
-pandas
-matplotlib
-IPython
-math
-torch
-scikit-learn
-tqdm

Additionally, the LSTM can run on a CPU, but is optimized for a machine with a CUDA-enabled GPU or Apple MPS.

## Usage
You can simply change out the file path to the data at the beginning of section 1.2. There may be errors with names of model parameters, but those are easy to fix.
