import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from torchinfo import summary



class LSTM1(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Define each LSTM layer individually to avoid sequential encapsulation issues

        self.transformer = nn.Transformer()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, batch_first=True)
        self.relu1 = nn.ReLU()
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes, bias=True)
        )

    def forward(self, x):
        # Pass through first LSTM layer
        out, _= self.lstm1(x)
        out = self.relu1(out)
       
        # Pass through second LSTM layer
        out, _ = self.lstm2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = out[:,-1, :]
       
        prediction = self.fc(out)
        return prediction
    


def train_model(lstm, train_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr= .001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.01)
    epochs = 100
    loss_history = []
    for epoch in range(epochs):
        for data, targets in train_loader:
            x = data.to(device=device)
            y = targets.to(device=device)

            optimizer.zero_grad()
            predicted_y = lstm(x)
            loss = criterion(predicted_y, y)

            loss_history.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step(loss)
        print(f"Epoch: {epoch} Completed")

    return lstm, loss_history

def load_data_in_dictionary(data):
    '''This loads the data into a dictionary. each index of dictionary has a pair of watersheds, 
       upper and lower, and their features are merged in the for loop below'''
    num_networks = 0
    network_dict = {}
    broken_pairs = []
    for i in range(int((data['pair_id']).max())+1):
        try:
            downstream = data[(data['pair_id']== i) & (data['du'] == 'd')]
            upstream = data[(data['pair_id']== i) & (data['du'] == 'u')]
            #print(downstream)
            #print(upstream)
            if downstream.empty or downstream.isnull().values.any():
                print(i, " DS is empty")
                continue
            if upstream.empty or upstream.isnull().values.any():
                print(i, " US is empty")
                continue
            '''area_ratio = upstream.iloc[0]["Shape_Area"] / downstream.iloc[0]["Shape_Area"]
            if area_ratio > 1:
                print(i, " area ratio too large")
                continue
            if abs(downstream.iloc[-1]['ID'] - upstream.iloc[-1]['ID']) > 100:
                print(i, " DS and US too far apart")
                continue'''
            network = downstream.merge(upstream, on="time")
            network.drop(["x_x", "y_x", "pair_id_x", "x_y", "y_y", "pair_id_y"], axis=1, inplace=True)
            network_dict[num_networks] = network
            num_networks += 1
        except:
            broken_pairs.append(i)
            raise Exception
    return network_dict, broken_pairs, num_networks

def viz_networks(ibuc):
    fig, ax = plt.subplots()
    print('Network:', ibuc)
    print("Streamflow (downstream) mean:", np.round(network_dict[ibuc].streamflow_x.mean(),2))
    print("Streamflow (upstream) mean:", np.round(network_dict[ibuc].streamflow_y.mean(),2))

    network_dict[ibuc].loc[:2000,['streamflow_x']].plot(ax=ax, legend=False)
    network_dict[ibuc].loc[:2000,['streamflow_y']].plot(ax=ax, legend=False)
    ax.set_title('Streamflow')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Streamflow')

    ax.legend(["Downstream flow", "Upstream flow"])
    
    # plt.show()
    plt.close()

def make_data_loader(network_list):
    loader = {}
    np_seq_X = {}
    np_seq_y = {}
    for ibuc in network_list:
        df = network_dict[ibuc]
        scaler_in_i = scaler_in.transform(df.loc[:, lstm_inputs])
        scaler_out_i = scaler_out.transform(df.loc[:, lstm_outputs])
        
        n_samples = scaler_in_i.shape[0] - seq_length
        
        np_seq_X[ibuc] = np.zeros((n_samples, seq_length, n_input))
        np_seq_y[ibuc] = np.zeros((n_samples, n_output)) 

        for i in range(n_samples):
            t = i + seq_length
            np_seq_X[ibuc][i, :, :] = scaler_in_i[i:t, :]
            np_seq_y[ibuc][i, :] = scaler_out_i[t, :]

        ds = [torch.Tensor(np_seq_X[ibuc]), torch.Tensor(np_seq_y[ibuc])]
        loader[ibuc] = ds
    return loader, np_seq_X, np_seq_y

def fit_scaler():
    frames = [network_dict[ibuc].loc[:, lstm_inputs] for ibuc in networks_for_training]
    df_in = pd.concat(frames)   
    print(df_in.shape)
    scaler_in = StandardScaler()
    scaler_in.fit(df_in)

    frames = [network_dict[ibuc].loc[:, lstm_outputs] for ibuc in networks_for_training]
    df_out = pd.concat(frames)    
    print(df_out.shape)
    scaler_out = StandardScaler()
    scaler_out.fit(df_out)
    return scaler_in, scaler_out


def split_parameters():
    # create lists of network indices for each set based on the given network splits
    networks_for_training = list(range(0, n_networks_split['train'] + 1))
    networks_for_val = list(range(n_networks_split['train'] + 1, 
                                 n_networks_split['train'] + n_networks_split['val'] + 1))


    
    # organize the split parameters into separate lists for each set
    train_split_parameters = [networks_for_training]
    val_split_parameters = [networks_for_val]         #we are definitely missing out on the entire 80% of the dataset in validation

    return [train_split_parameters, val_split_parameters]


def concatanate_tensors(loader, training = True, offest_buckets = 0):
    if training == True:
        concatanated_tensor_x = loader[0][0]
        concatanated_tensor_y = loader[0][1]

        for i in range(len(loader) - 1):
            concatanated_tensor_x = torch.cat((concatanated_tensor_x, loader[i+1][0]))
            concatanated_tensor_y = torch.cat((concatanated_tensor_y, loader[i+1][1]))
        return concatanated_tensor_x, concatanated_tensor_y    
    if training == False:
        concatanated_tensor_x = loader[offest_buckets][0]
        concatanated_tensor_y = loader[offest_buckets][1]

        for i in range(offest_buckets, len(loader) + offest_buckets - 1):
            concatanated_tensor_x = torch.cat((concatanated_tensor_x, loader[i+1][0]))
            concatanated_tensor_y = torch.cat((concatanated_tensor_y, loader[i+1][1]))
        return concatanated_tensor_x, concatanated_tensor_y   

  


data = pd.read_csv("data/data1003.csv")
print(f"The number of features available to us: {data.head(0)}")
network_dict, num_networks, broken_pairs = load_data_in_dictionary(data)
print(f"The number of networks we have:\t{len(network_dict)}")
print(f"The number of data points in each network:\t{network_dict[0].shape[0]}")
print(f"The number of features in each network:\t{network_dict[0].shape[1]}")

lstm_inputs = ['precip_rate_x', 'APCP_surface_x', 'TMP_2maboveground_x', 
    'DSWRF_surface_x', 'DLWRF_surface_x', 'PRES_surface_x', 
    'UGRD_10maboveground_x', 'VGRD_10maboveground_x', 'SPFH_2maboveground_x', 
    'elevation_mean_x', 'slope_mean_x', 'Shape_Area_x',
    'impervious_mean_x', 'dksat_soil_layers_stag=1_x', 'streamflow_x',
    'precip_rate_y', 'APCP_surface_y', 'TMP_2maboveground_y',
    'DSWRF_surface_y', 'DLWRF_surface_y', 'PRES_surface_y',
    'UGRD_10maboveground_y', 'VGRD_10maboveground_y', 'SPFH_2maboveground_y',
    'elevation_mean_y', 'slope_mean_y', 'Shape_Area_y',
    'impervious_mean_y', 'dksat_soil_layers_stag=1_y']

n_input = len(lstm_inputs)

lstm_outputs = ['streamflow_y']     #this is what we will be trying to predict using the LSTM model
n_output = len(lstm_outputs)
print(f"The number of features for LSTM model:{n_input}")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA device:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple M3/M2/M1 (Metal) device")
else:
    device = 'cpu'
    print("Using CPU")
# hidden_state_size = ( n_input + n_output ) * 3   I've commented this out because I will arbitrarily assign it later inside the LSTM1 class definition
# num_layers = 1
num_epochs = 5
batch_size = 256
seq_length = 24
# learning_rate = np.linspace(start=0.0001, stop=0.00001, num=num_epochs)   better adding a learning rate scheduler
n_networks = len(network_dict)

n_networks_split = {"train": (math.floor(n_networks * 0.7)), "val": math.floor(n_networks * 0.3)}
print(n_networks_split['val'])

[[networks_for_training], [networks_for_val]] = split_parameters()
print(networks_for_training)
print(networks_for_val)




    

displayed = 0

for ibuc in networks_for_training:
    viz_networks(ibuc)
    displayed += 1


for ibuc in networks_for_val:
    viz_networks(ibuc)
    displayed += 1



model = LSTM1(input_size=n_input, num_classes=n_output).to(device)
scaler_in, scaler_out = fit_scaler()

train_loader, np_train_seq_X, np_train_seq_y = make_data_loader(networks_for_training)
val_loader, np_val_seq_X, np_val_seq_y = make_data_loader(networks_for_val)
train_data_x, train_data_y = concatanate_tensors(train_loader)
val_data_x, val_data_y = concatanate_tensors(val_loader, False, len(train_loader))
train_dataset = torch.utils.data.TensorDataset(train_data_x, train_data_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size)
model, results = train_model(model, train_loader)