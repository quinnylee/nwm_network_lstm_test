import pandas as pd
import xarray as xr
import multiprocessing as mp
import glob
from exactextract import exact_extract
import datetime
import os
import multiprocessing.pool as mp2

# Defining classes to make nested parallelism work
class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(mp.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(mp2.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)

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

def chrtout_process(time_arg, comids):
    # process CHANNEL ROUTING (CHRTOUT) files, returns df of streamflow values
    #problem_comids = []

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
        try:
            forcing_data.append(zonal_stats['mean'][0])
        except:
            forcing_data.append(-9999)
            continue
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
    #lat = catchments['y'][k]
    #lon = catchments['x'][k]
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
    
    with mp.Pool(processes=10) as pool:
        results = pool.starmap(process_time, args)

    #print("finished mp")
    df = {t: row for t, row in results}
    df = pd.DataFrame.from_dict(df, orient='index', columns=colnames)
    return df

# Uses nested parallelism
def save_data_np(time_arg, catchments, comids, attr_paths, times, colnames, 
              hydrofabric, forcing_vars, forc_dataset, q_dataset):
    # Collects and saves data for all catchments over all timesteps
    # Outputs a file that logs errors and progress
    exp_dirname = '../runs/experiment_'+datetime.date.today().isoformat()+'/'
    if not os.path.exists(exp_dirname):
        os.makedirs(exp_dirname)

    output_name = exp_dirname + 'output.txt'
    t1 = datetime.datetime.now()
    # separate runs in output file
    with open(output_name, 'a') as file:
        toprint = ('MP run for ' + time_arg + ' started at ' + 
                   t1.strftime("%m/%d/%Y %H:%M:%S") + '\n')
        file.write(toprint)

    #args = [i for i in range(len(catchments))]
    args2 = [(j, catchments, comids, attr_paths, times, colnames, hydrofabric,
             forcing_vars, forc_dataset, q_dataset) for j in range(2)]

    pool = MyPool(10)

    results = pool.starmap(process_catchment, args2)
    results = pd.concat(results)

    try:
        parsed_time = time_arg.replace("/", "").strip("*")
        results.to_csv(exp_dirname + parsed_time + '.csv') 
        with open(output_name, 'a') as file:
            file.write('saved :D\n')
    except Exception as e:
        with open(output_name, 'a') as file:
            file.write(f"Error: {e}\n")
    pool.close()
    pool.join()

    t2 = datetime.datetime.now()
    with open(output_name, 'a') as file:
        toprint = ('MP run for ' + time_arg + ' elapsed time in s: ' 
                   + str((t2-t1).total_seconds()) + '\n')
        file.write(toprint)

# Regular file saving, no nested parallelism
def save_data_reg(time_arg, catchments, comids, attr_paths, times, colnames, 
              hydrofabric, forcing_vars, forc_dataset, q_dataset):
    # Collects and saves data for all catchments over all timesteps
    # Outputs a file that logs errors and progress
    exp_dirname = '../runs/experiment_'+datetime.date.today().isoformat()+'/'
    if not os.path.exists(exp_dirname):
        os.makedirs(exp_dirname)

    output_name = exp_dirname + 'output.txt'

    t1 = datetime.datetime.now()
    # separate runs in output file
    with open(output_name, 'a') as file:
        toprint = ('Regular run for ' + time_arg + ' started at ' 
                   + t1.strftime("%m/%d/%Y %H:%M:%S") + '\n')
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
        parsed_time = time_arg.replace("/", "").strip("*")
        data.to_csv(exp_dirname + parsed_time + '.csv')
        with open(output_name, 'a') as file:
            file.write('saved :D\n')
    except Exception as e:
        with open(output_name, 'a') as file:
            file.write(f"Error: {e}\n")

    t2 = datetime.datetime.now()
    with open(output_name, 'a') as file:
        toprint = ('Regular run for ' + time_arg +' elapsed time in s: ' + 
                   str((t2-t1).total_seconds()) + '\n')
        file.write(toprint)    