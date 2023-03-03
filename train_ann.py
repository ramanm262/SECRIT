import matplotlib.colors
import matplotlib.pyplot as plt
import datetime as dt
from sec import *

import preprocessing
from pickle import dump, load

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential  # May need to make it `tf.python.keras` depending on environment
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

# Setup
mode = "loading"  # training or loading

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

# Load the training data and prepare it in the correct format
syear, eyear = 2008, 2012
lead, recovery = 12, 24

# Load OMNI data
print("Loading OMNI data...")
omni_data = pd.read_feather(f"/data/ramans_files/omni-feather/omniData-{syear}-{eyear}-interp-None.feather")
omni_data = omni_data.rename(columns={"Epoch": "Date_UTC"})
omni_data.set_index("Date_UTC", inplace=True, drop=True)
omni_params = ["B_Total", "BX_GSE", "BY_GSM", "BZ_GSM", "flow_speed",
               "Vx", "Vy", "Vz", "proton_density", "T", "Pressure", "E_Field"]
omni_data = omni_data[omni_params]

delayed_params = ["B_Total", "BX_GSE", "BY_GSM", "BZ_GSM", "flow_speed",
                  "Vx", "Vy", "Vz", "proton_density", "T", "Pressure", "E_Field"]


def create_delays(df, name, time=20):
    for delay in np.arange(1, int(time) + 1):
        df[name + '_%s' % delay] = df[name].shift(delay).astype('float32')


n_data, e_data = pd.DataFrame([]), pd.DataFrame([])
print("Loading SuperMAG data...\n")
for station_name in tqdm.tqdm(stations_list):
    this_n_data, this_e_data = preprocessing.load_supermag(station_name, syear, eyear, data_path=supermag_data_path)
    n_data = pd.concat([n_data, this_n_data], axis=1)
    e_data = pd.concat([e_data, this_e_data], axis=1)

Z_matrix = [0]*2*len(n_data.columns)
station_num = 0
while station_num < len(n_data.columns):
    Z_matrix[2*station_num] = n_data.iloc[:, station_num]
    Z_matrix[2*station_num+1] = e_data.iloc[:, station_num]
    station_num += 1
Z_matrix = pd.concat(Z_matrix, axis=1)

# Create an all-data dataframe so that NaNs and storms can be dropped while keeping indices aligned
all_data = pd.concat([Z_matrix, omni_data], axis=1)
del omni_data, Z_matrix

print(f"Dropping NaNs from temporarily-combined SuperMAG and OMNI data.\nLength before dropping: "
      f"{len(all_data)}")
all_data.dropna(axis=0, inplace=True)
print(f"Length after dropping: {len(all_data)}\n")

# Split storm list into training, validation, and test storms
storm_list = pd.read_csv("stormList.csv", header=None, names=["dates"])
# Keep only storm-time training data
storm_data = []  # Will be a list of dataframes, each one corresponding to one storm
for date in tqdm.tqdm(storm_list["dates"], desc=f"Selecting storms"):
    stime = (dt.datetime.strptime(date, '%m-%d-%Y %H:%M')) - pd.Timedelta(hours=lead)  # Storm onset time
    etime = (dt.datetime.strptime(date, '%m-%d-%Y %H:%M')) + pd.Timedelta(hours=recovery)  # Storm end time
    this_storm = all_data[(all_data.index >= stime) & (all_data.index <= etime)]
    if len(this_storm) > 0:
        storm_data.append(this_storm)
del all_data
storm_data = pd.concat(storm_data, axis=0)

omni_data = storm_data[omni_params]
Z_matrix = storm_data.drop(columns=omni_params)
del storm_data

print("Generating SEC coefficients")
sec_data = gen_current_data(Z_matrix, station_coords_list, sec_coords_list, epsilon=1e-3)
target_scale = 1e6
sec_data = sec_data/target_scale
X_train, X_valid, y_train, y_valid = train_test_split(omni_data, sec_data, test_size=0.15, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
test_timestamp_vector = X_test.index

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
dump(scaler, open(f"models/scaler-ann-{syear}-{eyear}.pkl", "wb"))  # Save scaler

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
predictions = ann_model.predict(X_test)
for system in range(10):  # Should normally be in range(len(n_sec_lon*n_sec_lat))
    test_predictions = [timestamp[system]*target_scale for timestamp in predictions]
    ground_truth = y_test.iloc[:, system]*target_scale
    case.append(f"system_{system}")
    rmse.append(np.sqrt(mean_squared_error(ground_truth, test_predictions)))
    expv.append(explained_variance_score(ground_truth, test_predictions))
    r2.append(r2_score(ground_truth, test_predictions))
    corr.append(np.sqrt(r2_score(ground_truth, test_predictions)))

    plt.figure(figsize=(10, 6))
    plt.hist2d(ground_truth, test_predictions, bins=100)
    plt.xlabel("True")
    plt.ylabel(f"Predicted")
    plt.title(f"Real vs. Predicted coefficient for SEC #{system}")
    plt.savefig(f"plots/ANN-test-SEC{system}.png")

scores = pd.DataFrame({'case': case,
                       'rmse': rmse,
                       'expv': expv,
                       'r2': r2,
                       'corr': corr})
scores.to_csv(f"ann_scores-{syear}-{eyear}.csv", index=False)
