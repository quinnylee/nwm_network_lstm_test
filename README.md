# NWM Catchment Network LSTM: Predicting Upstream Flows
Branch: quinn_ml

## Description of repository

Our project’s goal is to construct a dataset of historical gage estimates for ungaged basins across the NextGen domain using machine learning (ML) techniques to assimilate known gage values from nearby basins into our predictions. ML models have been shown to predict flows well in ungaged basins (Kratzert et al., 2019; see also Ghaneei & Moradkhani, 2025; Frame et al., 2025), and if we constrain the ML training with information from observed flows in the same network, we can increase the accuracy of the ungaged flow estimates (Fisher et al., 2020). 

We have conducted experiments with synthetic hydrology to show that, with two-basin networks, one downstream and one upstream, an ML model can be trained to provide a higher accuracy estimate of the upstream flows from “ungaged” synthetic basins than can be obtained by training an ML model on individual basins alone [(Ramirez Molina et al., 2024; ](https://github.com/aarm1978/Synthetic_Stream_Gauges) [Lee et al., 2024)](https://github.com/quinnylee/synthetic_stream_gages). Building on these synthetic experiments, we propose to expand this proof-of-concept to include more robust training using National Water Model (NWM) data.

This project uses data from the [NWM 3.0 retrospective dataset](https://github.com/NOAA-Big-Data-Program/bdp-data-docs/blob/main/nwm/README.md) instead of synthetic data. We used the NWM 3.0 channel hydrofabric to select adjacent basin pairs (one upstream and one downstream). We used PyTorch to create a long short-term memory (LSTM) model that takes meteorological forcings, basin attributes, and downstream streamflow values to predict our "ungaged" upstream value. We chose to work with the NWM 3.0 retrospective dataset because it gave us a near-infinite amount of data to train and validate our model, and because we had greater control over the study area and network geometries. This phase of the project will result in a prototype workflow and LSTM model for ML-assisted data assimilation. 

Phases 3 and 4 of this project will be addressed in later repositories and will use real-world gaged data. These phases will result in retrospective datasets of high-confidence estimates of flow at ungaged basins. These datasets can be used by the hydrologic community as robust evaluation datasets to tune predictive models for ungaged basins within the NextGen Framework. This workflow will also lay the foundation for future researchers interested in ML-assisted data assimilation across more complex network topologies.

This model is still actively under development.

## Description of branch
This branch was used to develop a PyTorch LSTM model using torch Dataset and DataLoader objects to load model data into memory efficiently. These objects were used because loading all data into memory would cause out-of-memory issues.

## Installation and usage

You can simply clone this repository to access the LSTM, which is in the notebook, and a preliminary dataset.

The LSTM is located in data_train.ipynb, and one year's worth of data for around 500 randomly selected basin pairs in the continental United States are located in BIG_DATA as Parquet files. The model notebook is already configured to accept the Parquet files as inputs.

The following packages are required:
- numpy
- pandas
- matplotlib
- IPython
- math
- torch
- scikit-learn
- tqdm
- os
- warnings

Additionally, the LSTM can run on a CPU, but is optimized for a machine with a CUDA-enabled GPU or Apple MPS.

## Questions?

Contact Quinn Lee (author of branch) at qylee@ua.edu or Sonam Lama (author of model) at slama@crimson.ua.edu.

## References

Fisher, C. K., Pan, M., & Wood, E. F. (2020). Spatiotemporal assimilation–interpolation of discharge records through inverse streamflow routing. *Hydrology and Earth System Sciences*, 24(1), 293–305. https://doi.org/10.5194/hess-24-293-2020 

Frame, J. M., Araki, R., Bhuiyan, S. A., Bindas, T., Rapp, J., Bolotin, L., et al. (2025). Machine learning for a heterogeneous water modeling framework. *JAWRA Journal of the American Water Resources Association*, 61(1). https://doi.org/10.1111/1752-1688.70000 

Ghaneei, P., & Moradkhani, H. (2025). DeepBase: A deep learning-based daily Baseflow dataset across the United States. *Scientific Data*, 12(1). https://doi.org/10.1038/s41597-025-04389-y 

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. (2019). Toward improved predictions in ungauged basins: Exploiting the power of machine learning. *Water Resources Research*, 55(12), 11344–11354. https://doi.org/10.1029/2019wr026065  

Ramírez Molina, A. A., Frame, J., Halgren, J., & Gong, J. (2024). *Synthetic stream gauges: An LSTM-based approach to enhance river streamflow predictions in unmonitored segments*. ms, The University of Alabama. 
