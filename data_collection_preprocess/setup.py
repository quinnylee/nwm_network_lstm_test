import warnings
import pandas as pd
import os
import multiprocessing as mp
import geopandas as gpd

def read_cats(cats_path):
    # Reads information about desired catchments from csv
    cats_path = 'all_data.csv'
    catchments = pd.read_csv(cats_path)
    catchments = catchments[:1000]

    # generate pair IDs for each upstream/downstream pair of catchments in case
    # data gets shuffled around by accident
    pairids = []
    for i in range(len(catchments)):
        if i % 2 == 0:
            pairids.append(i / 2)
        else:
            pairids.append((i-1)/2)

    catchments['pair_id'] = pairids

    return catchments

def get_comids(catchments):
    # create list of COMIDs
    comids = catchments['comid'].tolist()

    return comids

# generate path to attribute files for one catchment, as well as CATID
def getclosest_array(i, catchments, ngiab_output_dir):
    catid = catchments.iat[i, 0]
    
    if catchments.iat[i,1] == 'd':
        attr_path = os.path.join(ngiab_output_dir, catid, 'config', 
                                 'cfe_noahowp_attributes.csv') 
    else:
        ds_catid = catchments.iat[i-1,0]
        attr_path = os.path.join(ngiab_output_dir, ds_catid, 'config', 
                                 'cfe_noahowp_attributes.csv')
    
    attr_df = pd.read_csv(attr_path)
    attr_df.set_index('divide_id', inplace=True)

    return attr_path, catid

def hydrofabric_setup(hf_path):
    # Set up hydrofabric
    # change path to your hydrofabric download location

    hydrofabric = gpd.read_file(hf_path)

    # reproject to match forcing dataset CRS
    hydrofabric = hydrofabric.to_crs("+proj=lcc +units=m +a=6370000.0 " + 
                                    "+b=6370000.0 +lat_1=30.0 +lat_2=60.0 " +
                                    "+lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 " +
                                    "+k_0=1.0 +nadgrids=@null +wktext +no_defs")
    return hydrofabric

# generates attribute filepaths and CATIDs for all catchments in parallel
def parallel_gca(catchments, ngiab_output_dir):
    with mp.Pool() as pool:
        result = pool.starmap(getclosest_array, 
                              [(i, catchments, ngiab_output_dir) 
                               for i in range(len(catchments))])
    attr_paths, catids = zip(*result)
    return attr_paths, catids

def col_names(attr_paths, forcing_vars):
    # define column names for final dataset
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

    return colnames
