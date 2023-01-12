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
import pickle
from preprocessing import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import precision_recall_curve, f1_score, auc

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import load_model
import tensorflow as tf


mode = "training"  # "training" or "loading"

omni_data_path = "/data/ramans_files/omni-feather/"
supermag_data_path = "/data/ramans_files/mag-feather/"
iono_data_path = "/data/ramans_files/iono-feather/"

# CNN hyperparameters
time_history = 60  # Minutes of time history to train on
epochs = 100  # Maximum number of training epochs
conv_filters_list = [16]  # List whose elements are the number of filters in the output of the corresponding conv layer
fc_nodes_list = [128, 64]  # List whose elements are the number of nodes in each FC layer (NOT including output layer)
init_lr = 1e-6  # Initial learning rate
dropout_rate = 0.2

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
n_sec_lat, n_sec_lon = 5, 10
w_lon, e_lon, s_lat, n_lat = 210., 300., 35., 65.
sec_coords_list = [np.linspace(s_lat, n_lat, n_sec_lat), np.linspace(w_lon, e_lon, n_sec_lon)]


omni_data, n_data, e_data, sec_data = preprocess_data(2008, 2008, stations_list, station_coords_list, sec_coords_list,
                                                      omni_data_path, supermag_data_path, iono_data_path,
                                                      calculate_sec=False,)



class CNN(Sequential):
    def __init__(self, conv_filters_list, fc_nodes_list, n_features, loss="mse", dropout_rate=0.2):
        super(CNN, self).__init__()  # Call parent class' constructor

        # Convolutional segment
        for conv_layer in range(len(conv_filters_list)):
            self.add(Conv2D(conv_filters_list[conv_layer], (1,2), padding="same", activation="relu",
                            input_shape=(time_history, n_features, 1)))
            self.add(MaxPooling2D())
        self.add(Flatten())

        # Fully-connected segment
        for fc_layer in range(len(fc_nodes_list)):
            self.add(Dense(fc_nodes_list[fc_layer], activation="relu"))
            self.add(Dropout(dropout_rate))

        # Output layer
        self.add(Dense(1, activation="relu"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
        self.compile(optimizer=optimizer, loss=loss)

    def early_stop(self, early_stop_patience=25):
        return EarlyStopping(monitor="val_loss", verbose=1, patience=early_stop_patience)

if mode == "training":
    model = CNN(conv_filters_list, fc_nodes_list, n_features=13, dropout_rate=0.2)
    early_stop = model.early_stop(early_stop_patience=25)

elif mode == "loading":

else:
    raise ValueError("mode must be 'training' or 'loading'")

