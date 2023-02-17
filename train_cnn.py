import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import inspect
import time
import matplotlib.dates as mdates
import random
from pickle import dump, load
from preprocessing import *
from nn import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import precision_recall_curve, f1_score, auc


mode = "training"  # "training" or "loading"

omni_data_path = "/data/ramans_files/omni-feather/"
supermag_data_path = "/data/ramans_files/mag-feather/"
iono_data_path = "/data/ramans_files/iono-feather/"

# CNN hyperparameters
time_history = 30  # Minutes of time history to train on
epochs = 100  # Maximum number of training epochs
early_stop_patience = 10 # Number of epochs to continue to train while validation loss does not improve
conv_filters_list = [128]  # List whose elements are the number of filters in the output of the corresponding conv layer
fc_nodes_list = [1000, 250]  # List whose elements are the number of nodes in each FC layer (NOT including output layer)
init_lr = 1e-5  # Initial learning rate
dropout_rate = 0.2
valid_proportion = 0.25  # Proportion of the data to be assigned to validate the model during training

# SEC hyperparameters
stations_list = ['YKC', 'CBB', 'BLC', 'SIT', 'BOU', 'VIC', 'NEW', 'OTT', 'FRD', 'GIM', 'FCC', 'FMC', 'FSP',
                 'SMI', 'ISL', 'PIN', 'RAL', 'INK', 'CMO', 'IQA', 'LET',
                 'T16', 'T32', 'T33', 'T36']
station_coords_list = [np.array([62.48, 69.1, 64.33, 57.07, 40.13, 48.52, 48.27, 45.4, 38.2, 56.38, 58.76, 56.66,
                                 61.76, 60.02, 53.86, 50.2, 58.22, 68.25, 64.87, 63.75, 49.64, 39.19, 49.4, 54.0,
                                 54.71]),
                       np.array([245.52, 255.0, 263.97, 224.67, 254.77, 236.58, 242.88, 284.45, 282.63, 265.36,
                                 265.92, 248.79, 238.77, 248.05, 265.34, 263.96, 256.32, 226.7, 212.14, 291.48,
                                 247.13, 240.2, 277.7, 259.1, 246.69])]
n_sec_lat, n_sec_lon = 4, 11
w_lon, e_lon, s_lat, n_lat = 210., 300., 35., 65.
sec_coords_list = [np.linspace(s_lat, n_lat, n_sec_lat), np.linspace(w_lon, e_lon, n_sec_lon)]

# Load the training data and prepare it in the correct format
syear, eyear = 2008, 2012
omni_data, sec_data, n_data, e_data = preprocess_data(syear, eyear, stations_list, station_coords_list, sec_coords_list,
                                                      omni_data_path, supermag_data_path, iono_data_path,
                                                      calculate_sec=False, test_proportion=0.2)

if mode == "training":
    # Instantiate the model
    model = CNN(conv_filters_list, fc_nodes_list, n_features=12, time_history=time_history,
                output_nodes=n_sec_lat*n_sec_lon, init_lr=init_lr, dropout_rate=dropout_rate)
    early_stop = model.early_stop(early_stop_patience=early_stop_patience)

    sec_data = sec_data/1e6
    print(sec_data.head())
    X_train_df, X_valid_df, y_train, y_valid = \
        train_test_split(omni_data, sec_data, test_size=valid_proportion, shuffle=True)

    X_train_df = X_train_df.reset_index(drop=True).to_numpy()
    X_valid_df = X_valid_df.reset_index(drop=True).to_numpy()

    print("Scaling data...")
    scaler = StandardScaler()
    scaler.fit(X_train_df)
    X_train_df = scaler.transform(X_train_df)
    X_valid_df = scaler.transform(X_valid_df)
    dump(scaler, open(f"scalers/scaler_{syear}-{eyear}.pkl", "wb"))  # Save scaler

    # Split up the training data into batches
    X_train, X_valid = [], []
    for batch in tqdm.trange(len(X_train_df) - time_history, desc="Preparing batches of training data"):
        X_train.append(X_train_df[batch:batch + time_history])
    y_train = y_train.iloc[time_history:].to_numpy()  # This method predicts 1 minute ahead
    for batch in tqdm.trange(len(X_valid_df) - time_history, desc="Preparing batches of validation data"):
        X_valid.append(X_valid_df[batch:batch + time_history])
    y_valid = y_valid.iloc[time_history:].to_numpy()
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)

    # Reshape the training data, so it is accommodated by the input layer
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], 1))

    model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
              verbose=1, shuffle=True, epochs=epochs,
              callbacks=[early_stop])

    model.save(f"models/SECRIT_{syear}-{eyear}.h5")

    # Plot the training loss curve
    history = model.history  # Save the model loss history
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title(f"Training Loss {syear}-{eyear}")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')
    plt.savefig(f"plots/training_loss_{syear}-{eyear}.png")

elif mode == "loading":
    pass

else:
    raise ValueError("mode must be 'training' or 'loading'")
