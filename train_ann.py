import matplotlib.colors
import matplotlib.pyplot as plt
import datetime as dt
import glob
from sec import *

import preprocessing
from pickle import dump, load

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential  # May need to make it `tf.python.keras` depending on environment
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

# Used to suppress annoying commandline complaints about repeatedly inserting columns into a DataFrame
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None


# Setup
mode = "training"  # training or loading the ANN
calculate_sec = True # Whether to calculate or load SEC coefficients

omni_data_path = "/data/ramans_files/omni-feather/"
supermag_data_path = "/data/ramans_files/mag-feather/"
iono_data_path = "/data/ramans_files/iono-feather/"

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

# Other hyperparameters
time_history = 30
target_scale = 1e5
proportions = [0.7, 0.15, 0.15]  # Train, validation, and test set proportions, respectively

# Load the training data and prepare it in the correct format
syear, eyear = 2008, 2012
lead, recovery = 12, 24

# Load OMNI data
print("Loading OMNI data...\n")
omni_data = preprocessing.load_omni(syear=syear, eyear=eyear, data_path="/data/ramans_files/omni-feather/")
omni_params = ["B_Total", "BX_GSE", "BY_GSM", "BZ_GSM", "flow_speed",
               "Vx", "Vy", "Vz", "proton_density", "T", "Pressure", "E_Field"]
delayed_params = ["B_Total", "BX_GSE", "BY_GSM", "BZ_GSM", "flow_speed",
                  "Vx", "Vy", "Vz", "proton_density", "T", "Pressure", "E_Field"]  # Usually the same as omni_params

# Load SuperMAG data
n_data, e_data = pd.DataFrame([]), pd.DataFrame([])
print("Loading SuperMAG data...\n")
for station_name in tqdm.tqdm(stations_list):
    this_n_data, this_e_data = preprocessing.load_supermag(station_name, syear, eyear, data_path=supermag_data_path)
    n_data = pd.concat([n_data, this_n_data], axis=1)
    e_data = pd.concat([e_data, this_e_data], axis=1)

if calculate_sec:
    # Initialize a matrix whose rows are timesteps and columns are Z vectors described in Amm & Viljanen 1999
    Z_matrix = [0]*2*len(n_data.columns)
    station_num = 0
    while station_num < len(n_data.columns):
        Z_matrix[2*station_num] = n_data.iloc[:, station_num]
        Z_matrix[2*station_num+1] = e_data.iloc[:, station_num]
        station_num += 1
    Z_matrix = pd.concat(Z_matrix, axis=1)
    mag_params = Z_matrix.columns.values

    # Create an all-data dataframe so that NaNs and storms can be dropped while keeping indices aligned
    all_data = pd.concat([Z_matrix, omni_data], axis=1)
    del omni_data, Z_matrix

    # Creating time history
    for param in tqdm.tqdm(delayed_params, desc=f"Creating time history"):
        preprocessing.create_delays(all_data, param, time=time_history)

    print(f"Dropping NaNs from temporarily-combined SuperMAG and OMNI data.\nLength before dropping: "
          f"{len(all_data)}")
    all_data.dropna(axis=0, inplace=True)
    print(f"Length after dropping: {len(all_data)}\n")

    # Split storm list into training, validation, and test storms
    storm_list = pd.read_csv("stormList.csv", header=None, names=["dates"])
    train_storm_list, valid_storm_list, test_storm_list = preprocessing.split_storm_list(storm_list, proportions=proportions)

    # Keep only storm-time training data
    all_storm_data = []  # Its elements are three lists of dataframes, one for each split of the dataset
    for sublist, status in \
            zip([train_storm_list, valid_storm_list, test_storm_list], ["training", "validation", "test"]):
        subdata = preprocessing.get_storm_data(all_data, sublist, lead, recovery, status)  # List of storm-time dataframes
        all_storm_data.append(subdata)
    del all_data

    split_sec_data, split_omni_data = [], []  # Each is a list of lists of storm-time dataframes
    for dataset, status in zip(all_storm_data, ["training", "validation", "test"]):
        print(f"Generating SEC coefficients for {status} set. "
              "To load them from a file instead, use 'calculate_sec=False'")
        this_sec_dataset, this_omni_dataset = [], []  # Each is a list of storm-time dataframes for a certain split
        storm_num = 0
        for storm_df in tqdm.tqdm(dataset):
            this_Z_matrix = storm_df[mag_params]
            this_omni_data = storm_df.drop(columns=mag_params)
            this_omni_dataset.append(this_omni_data)

            this_sec_data = gen_current_data(this_Z_matrix, station_coords_list, sec_coords_list, epsilon=1e-3,
                                             disable_tqdm=True)
            reduced_index = this_Z_matrix.index
            this_sec_data.index = reduced_index
            this_sec_data.columns = this_sec_data.columns.astype(str)  # Column names must be strings to save to feather
            this_sec_dataset.append(this_sec_data)
            this_sec_data.reset_index().to_feather(f"{iono_data_path}storms/I_{syear}-{eyear}_{status}-{storm_num}.feather")
            storm_num += 1
        split_sec_data.append(this_sec_dataset)
        split_omni_data.append(this_omni_dataset)
else:
    split_sec_data, split_omni_data = [], []
    for status in ["training", "validation", "test"]:
        this_sec_dataset, this_omni_dataset = [], []
        print(f"Loading {status} SEC coefficients for {status} dataset\n")
        for storm_file in glob.glob(f"{iono_data_path}storms/I_{syear}-{eyear}_{status}-*.feather"):
            this_sec_data = pd.read_feather(storm_file)
            this_sec_data = this_sec_data.rename(columns={"index": "Date_UTC"})
            this_sec_data.set_index("Date_UTC", inplace=True, drop=True)
            reduced_index = this_sec_data.index
            this_omni_data = omni_data.loc[reduced_index]
            this_sec_dataset.append(this_sec_data)
            this_omni_dataset.append(this_omni_data)
        split_sec_data.append(this_sec_dataset)
        split_omni_data.append(this_omni_dataset)

# Cut down the N and E component observations to the same period as the test-set SEC coefficients
to_index = pd.concat(split_sec_data[2], axis=0)
test_n_data = n_data.loc[to_index.index]
test_e_data = e_data.loc[to_index.index]

X_train_storms, X_valid_storms, X_test_storms, y_train_storms, y_valid_storms, y_test_storms = \
    split_omni_data[0], split_omni_data[1], split_omni_data[2], split_sec_data[0], split_sec_data[1], split_sec_data[2]

# For ANN, don't need to keep storms separate except for test set
X_train = pd.concat(X_train_storms, axis=0)
y_train = pd.concat(y_train_storms, axis=0)
X_valid = pd.concat(X_valid_storms, axis=0)
y_valid = pd.concat(y_valid_storms, axis=0)

y_train, y_valid = y_train/target_scale, y_valid/target_scale


print("Scaling data...")
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train.reset_index(drop=True))
X_valid = scaler.transform(X_valid.reset_index(drop=True))
dump(scaler, open(f"scalers/ann_scaler_{syear}-{eyear}.pkl", "wb"))  # Save scaler

if mode == "training":

    ann_model = Sequential()

    ann_model.add(Dense(500, activation='relu', input_shape=(X_train.shape[1],)))
    ann_model.add(Dropout(0.2))
    ann_model.add(Dense(250, activation='relu'))
    ann_model.add(Dropout(0.2))
    ann_model.add(Dense(125, activation='relu'))
    ann_model.add(Dropout(0.2))
    ann_model.add(Dense(62, activation='relu'))
    ann_model.add(Dense(n_sec_lon*n_sec_lat))

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)  # Use to control learning rate.
    ann_model.compile(optimizer="adam", loss="mse")
    early_stop = EarlyStopping(monitor="val_loss", verbose=1, patience=25)

    print(20 * '#' + f"\nTraining model...\n" + 20 * '#')

    ann_model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=500,
                  callbacks=[early_stop], shuffle=True, verbose=True)
    ann_model.save(f"models/ANN_SEC_{syear}-{eyear}.h5")

    losses_training_model = pd.DataFrame(ann_model.history.history)
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    plt.plot(losses_training_model["loss"], label="Training loss")
    plt.plot(losses_training_model["val_loss"], label="Testing loss")
    plt.title(f"Loss Functions for Training, ANN version")
    ax.set_ylabel("Loss (MSE)")
    ax.set_xlabel("Training Epoch")
    plt.legend()
    plt.savefig("./plots/loss-functions-ANN-SEC.png")

elif mode == "loading":
    ann_model = tf.keras.models.load_model(f"models/ANN_SEC_{syear}-{eyear}.h5")

else:
    raise ValueError("The 'mode' variable must be set to 'training' or 'loading'")


# Testing
case, rmse, expv, r2, corr = [], [], [], [], []  # Initialize lists for scoring each model
print(f"Testing model...")
X_test, y_test = pd.concat(X_test_storms, axis=0), pd.concat(y_test_storms, axis=0)
y_test = y_test/target_scale
predictions = ann_model.predict(X_test)
for system in range(10):  # Should normally be in range(len(n_sec_lon*n_sec_lat))
    test_predictions = [timestamp[system]*target_scale for timestamp in predictions]
    ground_truth = y_test.iloc[:, system]*target_scale
    case.append(f"system_{system}")
    # rmse.append(np.sqrt(mean_squared_error(ground_truth, test_predictions)))
    # expv.append(explained_variance_score(ground_truth, test_predictions))
    # r2.append(r2_score(ground_truth, test_predictions))
    # corr.append(np.sqrt(r2_score(ground_truth, test_predictions)))

    plt.figure(figsize=(10, 6))
    plt.hist2d(ground_truth, test_predictions, bins=100, range=[[-1e7, 1e7], [-1e7, 1e7]])
    plt.plot(np.linspace(-1e7, 1e7, 200), np.linspace(-1e7, 1e7, 200), c="orange")
    plt.xlabel("True")
    plt.ylabel(f"Predicted")
    plt.title(f"Real vs. Predicted coefficient for SEC #{system}")
    plt.savefig(f"plots/ANN-density-SEC{system}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(400), test_predictions[:400], label="predicted")
    plt.plot(np.arange(400), ground_truth[:400], label="actual")
    plt.xlabel("Time")
    plt.ylabel(f"Coefficient (A)")
    plt.title(f"Real vs. Predicted for SEC #{system}")
    plt.savefig(f"plots/ANN-prediction-SEC{system}.png")


scores = pd.DataFrame({'case': case,
                       'rmse': rmse,
                       'expv': expv,
                       'r2': r2,
                       'corr': corr})
scores.to_csv(f"ann_scores-{syear}-{eyear}.csv", index=False)
