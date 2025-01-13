import pandas as pd
import xarray as xr
import multiprocessing as mp
import glob
from exactextract import exact_extract
import datetime
import os

def forcing_process(time_arg):
    # use globbed filepath, opens forcing NC files 
    forc_path = "/media/volume/Clone_Imp_Data/FORCING/" + time_arg + \
                ".LDASIN_DOMAIN1"
    forc_dataset = xr.open_mfdataset(forc_path)

    # get timestamps for all time steps in study period
    times = forc_dataset.time.values

    '''
    # get time strings
    time_strs = []
    for i in range(len(times)):
        pddt = pd.to_datetime(times[i])
        newstr = pddt.strftime("%Y-%m-%d %H:%M:%S")
        time_strs.append(newstr)'''

    #return forc_dataset, time_strs
    return forc_dataset, times

def chrtout_process(time_arg, comids):
    # process CHANNEL ROUTING (CHRTOUT) files, returns df of streamflow values
    #problem_comids = []

    # gets all relevant streamflow data from one timestep (one CHRTOUT file)
    def process_chrtout_file2(filename, id_list, value_col="streamflow"):
        with xr.open_dataset(filename) as ds:
            df =  ds[["time", value_col]].sel(feature_id=id_list).to_dataframe()
            # with ThreadPoolExecutor(max_workers = 100) as executor:
            #     df_list = list(executor.map(
            #                    lambda id: nested_multithreading_catchments(
            #                       id, ds, value_col), 
            #                    id_list))
            # df = pd.concat(df_list)
            return df
    
    '''
    # gets all relevant streamflow data from one timestep (one CHRTOUT file)
    def process_chrtout_file(filename, id_list, value_col="streamflow"):
        with xr.open_dataset(filename) as ds:
            df_list = []
            try:
                for id in id_list:
                    df_cat = ds[["time", value_col]].sel(
                        feature_id=id).to_dataframe()
                    df_list.append(df_cat)
            except:
                problem_comids.append(id) 
                print(id, " not found")
                #id_list.remove[id]

            df = pd.concat(df_list)
            return df'''

    # gets all relevant streamflow data from multiple timesteps through 
    # parallelization
    def get_q(qlat_files, id_list, index_col="feature_id", 
              value_col="streamflow"):

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(process_chrtout_file2, 
                                   [(filename, id_list) 
                                    for filename in qlat_files])

        frame = pd.concat(results, axis=0, ignore_index=False)

        return frame

    # use globbed filepath to generate q_dataset
    chrtout_path = "/media/volume/Clone_Imp_Data/CHRTOUT/" + time_arg + \
        ".CHRTOUT_DOMAIN1"
    q_dataset = get_q(glob.glob(chrtout_path), comids)
    q_dataset.reset_index(inplace=True)

    return q_dataset

# Collect forcings, attributes, and streamflow data for a catchment at one time 
# step
def process_time(t, comid, du, pair_id, catid, attr_values, hydrofabric,
                 forcing_vars, forc_dataset, q_dataset):

    # Initialize the row with basic catchment information
    row = [comid, catid, du, pair_id]

    forcing_data = []
    gdf = hydrofabric.loc[hydrofabric['divide_id']==catid]
    for var in forcing_vars:
        raster = forc_dataset.sel(time=t)[var]
        raster = raster.rio.write_crs("+proj=lcc +units=m +a=6370000.0 " \
                                      "+b=6370000.0 +lat_1=30.0 +lat_2=60.0 " \
                                      "+lat_0=40.0 +lon_0=-97.0 +x_0=0 " \
                                      "+y_0=0 +k_0=1.0 +nadgrids=@null " \
                                      "+wktext +no_defs")
        zonal_stats = exact_extract(raster, gdf, 'mean', 
                                    include_cols='divide_id', output='pandas')
        forcing_data.append(zonal_stats['mean'][0])
    # Append forcing data for each variable
    for var_value in forcing_data:
        row.append(var_value)

    # Streamflow data for the given time and comid
    filtered_data = q_dataset[(q_dataset['time']==t) & 
                              (q_dataset['feature_id']==comid)]
    streamflow_value = filtered_data['streamflow'].iloc[-1]
    row.append(streamflow_value)
    #print(t)

    # Append attributes (assuming these were preloaded)
    row.extend(attr_values)

    return (t, row)

# Collects forcings, attributes, streamflow data for one catchment over all 
# time steps 
def process_catchment(k, catchments, comids, attr_paths, times, colnames,
                      hydrofabric, forcing_vars, forc_dataset, q_dataset):
    df = {}
    lat = catchments['y'][k]
    lon = catchments['x'][k]
    comid = comids[k]
    attr_path = attr_paths[k]
    catid = catchments['catchment_id'][k]
    du = catchments['du'][k]
    pair_id = catchments['pair_id'][k]
    attr_df = pd.read_csv(attr_path)
    attr_df.set_index('divide_id', inplace=True)
    attr_values = attr_df.loc[catid].tolist()

    #print("loaded atts")

    args = [(times[i], comid, du, pair_id, catid, attr_values, hydrofabric,
             forcing_vars, forc_dataset, q_dataset)
            for i in range(len(times))]
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(process_time, args)

    #print("finished mp")
    df = {t: row for t, row in results}
    df = pd.DataFrame.from_dict(df, orient='index', columns=colnames)
    return df

def save_data(time_arg, catchments, comids, attr_paths, times, colnames, 
              hydrofabric, forcing_vars, forc_dataset, q_dataset):
    # Collects and saves data for all catchments over all timesteps
    # Outputs a file that logs errors and progress
    exp_dirname = '../runs/experiment_'+datetime.date.today().isoformat()+'/'
    if not os.path.exists(exp_dirname):
        os.makedirs(exp_dirname)

    output_name = exp_dirname + 'output.txt'

    # separate runs in output file
    with open(output_name, 'a') as file:
        toprint = 'Run started at ' + \
            datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ' for ' + \
            time_arg + '\n'
        file.write(toprint)

    results = []
    #for i in range(len(catchments)):
    for i in range(2):
        try:
            result = process_catchment(i, catchments, comids, attr_paths,
                                       times, colnames, hydrofabric,
                                       forcing_vars, forc_dataset, q_dataset)
            results.append(result)
            with open(output_name, 'a') as file:
                toprint = str(i) + ' done\n'
                file.write(toprint)
        except:
            with open(output_name, 'a') as file:
                toprint = str(i) + ' had an error\n'
                file.write(toprint)
            continue

    try:
        data = pd.concat(results)
        with open(output_name, 'a') as file:
            file.write('df concatenated\n')
    except Exception as e:
        with open(output_name, 'a') as file:
            file.write(f"Error: {e}\n")

    try:
        data.to_csv(exp_dirname + time_arg + '.csv')
        with open(output_name, 'a') as file:
            file.write('saved :D\n')
    except Exception as e:
        with open(output_name, 'a') as file:
            file.write(f"Error: {e}\n")