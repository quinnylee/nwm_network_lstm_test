# NWM Catchment Network LSTM: Predicting Upstream Flows
Branch: data-collection

## Table of Contents
- [NWM Catchment Network LSTM: Predicting Upstream Flows](#nwm-catchment-network-lstm-predicting-upstream-flows)
  - [Table of Contents](#table-of-contents)
  - [Description of repository](#description-of-repository)
  - [Description of branch](#description-of-branch)
  - [Installation and usage](#installation-and-usage)
  - [Directories and content](#directories-and-content)
    - [route\_link/01\_selectal](#route_link01_selectal)
    - [route\_link/02\_subset\_data](#route_link02_subset_data)
  - [Questions?](#questions)
  - [References/Credits](#referencescredits)

## Description of repository

Our project's goal is to construct a dataset of historical gage estimates for ungaged basins across the NextGen domain using machine learning (ML) techniques to assimilate known gage values from nearby basins into our predictions. ML models have been shown to predict flows well in ungaged basins (Kratzert et al., 2019; see also Ghaneei & Moradkhani, 2025; Frame et al., 2025), and if we constrain the ML training with information from observed flows in the same network, we can increase the accuracy of the ungaged flow estimates (Fisher et al., 2020). 

We have conducted experiments with synthetic hydrology to show that, with two-basin networks, one downstream and one upstream, an ML model can be trained to provide a higher accuracy estimate of the upstream flows from "ungaged" synthetic basins than can be obtained by training an ML model on individual basins alone [(Ramirez Molina et al., 2024; ](https://github.com/aarm1978/Synthetic_Stream_Gauges) [Lee et al., 2024)](https://github.com/quinnylee/synthetic_stream_gages). Building on these synthetic experiments, we propose to expand this proof-of-concept to include more robust training using National Water Model (NWM) data.

This project uses data from the [NWM 3.0 retrospective dataset](https://github.com/NOAA-Big-Data-Program/bdp-data-docs/blob/main/nwm/README.md) instead of synthetic data. We used the NWM 3.0 channel hydrofabric to select adjacent basin pairs (one upstream and one downstream). We used PyTorch to create a long short-term memory (LSTM) model that takes meteorological forcings, basin attributes, and downstream streamflow values to predict our "ungaged" upstream value. We chose to work with the NWM 3.0 retrospective dataset because it gave us a near-infinite amount of data to train and validate our model, and because we had greater control over the study area and network geometries. This phase of the project will result in a prototype workflow and LSTM model for ML-assisted data assimilation. 

Phases 3 and 4 of this project will be addressed in later repositories and will use real-world gaged data. These phases will result in retrospective datasets of high-confidence estimates of flow at ungaged basins. These datasets can be used by the hydrologic community as robust evaluation datasets to tune predictive models for ungaged basins within the NextGen Framework. This workflow will also lay the foundation for future researchers interested in ML-assisted data assimilation across more complex network topologies.

This model is still actively under development.

## Description of branch
The purpose of this branch is to develop a small-scale prototype input dataset for our ML model. We are limiting our study area to Alabama, but once the workflow is refined, we will expand our study area to the whole NWM domain. The NWM's RouteLink files and channel hydrofabric were used to select approximately 10,000 basin pairs in Alabama (one upstream and one downstream). 

The processes to effectively subset NWM forcings to the study area and period are incomplete and non-functional. The existing code draws heavily from the [NGIAB_data_preprocess package (Cunningham et al., 2025)](https://github.com/CIROH-UA/NGIAB_data_preprocess). **Our end goal with this branch is to use the selected basin pairs in `al_pairs.txt` to subset forcings, basin attributes, and streamflow values and format them in a Parquet file that is compatible with the models in main and quinn_ml. We would greatly appreciate any contributions to this area.**

## Installation and usage

You can simply clone this repository and checkout the `data-collection` branch to access the data processing tools in the `route_link` folder. Make sure to unzip the states .zip file sourced from the [U.S. Census Bureau](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) if you plan on using the tools in `route_link/01_selectal`.

The following packages are required to run the files in the `route_link` directory:
- fsspec
- xarray
- kerchunk
- geopandas
- functools
- numpy
- typing
- logging
- json
- time
- multiprocessing
- pandas
- exactextract
- pathlib
- rich
- psutil
- math
- warnings
- dask
- shutil
- os
- shapely

## Directories and content

### route_link/01_selectal

`route_link/01_selectal` contains tools to select basin pairs in Alabama by traversing the NWM 3.0 hydrofabric. The tools in this directory draw heavily from Halgren (2024)'s `route_link_fsspec.ipynb`.

The files in this directory are as follows:
- `nhd_network.py` from the [NOAA-OWP t-route repository](https://github.com/NOAA-OWP/t-route/)
  - Functions from this module are imported in the notebooks.
- `al_catchments.ipynb`
  - Subsets a `routelink.nc` file to a specific state.
- `find_reaches.ipynb`
  - Traverses a `routelink.nc` file to find networks of connected basins of particular sizes and shapes. This notebook is pre-configured to find two-basin (one upstream and one downstream) networks from the Alabama routelink file.
- `al_routelink.nc`
  - Subset of [`RouteLink_CONUS.nc`](https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/nwm.v3.0.13/parm/domain/) to Alabama using the included shapefile.
- `al_pairs.txt`
  - Dictionary of NWM reach IDs selected from `find_reaches.ipynb`. Keys are headwaters and values are tailwaters.

### route_link/02_subset_data
`route_link/02_subset_data` contains `incorporate_ngiab_pp.ipynb`, the notebook used to subset NWM forcings. **This is the notebook that requires the most code review and development. This notebook currently does not work.**

This branch assumes that you have a good amount of NetCDF files downloaded from the NOAA NWM retrospective AWS bucket and that you want to preprocess these files into a format that the LSTM model code likes. This branch also assumes that you have the NWM 3.0 hydrofabric geodatabase downloaded.

## Questions?

Contact Quinn Lee (author of branch) at qylee@ua.edu or Sonam Lama (author of model) at slama@crimson.ua.edu.

## References/Credits

Fisher, C. K., Pan, M., & Wood, E. F. (2020). Spatiotemporal assimilation–interpolation of discharge records through inverse streamflow routing. *Hydrology and Earth System Sciences*, 24(1), 293–305. https://doi.org/10.5194/hess-24-293-2020

Frame, J. M., Araki, R., Bhuiyan, S. A., Bindas, T., Rapp, J., Bolotin, L., et al. (2025). Machine learning for a heterogeneous water modeling framework. *JAWRA Journal of the American Water Resources Association*, 61(1). https://doi.org/10.1111/1752-1688.70000

Ghaneei, P., & Moradkhani, H. (2025). DeepBase: A deep learning-based daily Baseflow dataset across the United States. *Scientific Data*, 12(1). https://doi.org/10.1038/s41597-025-04389-y

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. (2019). Toward improved predictions in ungauged basins: Exploiting the power of machine learning. *Water Resources Research*, 55(12), 11344–11354. https://doi.org/10.1029/2019wr026065

Ramírez Molina, A. A., Frame, J., Halgren, J., & Gong, J. (2024). *Synthetic stream gauges: An LSTM-based approach to enhance river streamflow predictions in unmonitored segments*. ms, The University of Alabama.

James Halgren: conceptualization

Josh Cunningham: massive amounts of code