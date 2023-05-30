import pandas as pd

from sec import *
import datetime as dt
import glob


def load_omni(syear, eyear, data_path):
    omni_data = pd.read_feather(data_path + f"omniData-{syear}-{eyear}-interp-None.feather")
    omni_data = omni_data.rename(columns={"Epoch": "Date_UTC"})
    omni_data.set_index("Date_UTC", inplace=True, drop=True)
    omni_params = ["B_Total", "BX_GSE", "BY_GSM", "BZ_GSM", "flow_speed",
                   "Vx", "Vy", "Vz", "proton_density", "T", "Pressure", "E_Field"] # Only use these features
    omni_data = omni_data[omni_params]

    return omni_data


def load_supermag(station_name, syear, eyear, data_path):
    this_mag_data = pd.read_feather(data_path + f"magData-{station_name}_{syear}-"
                                    f"{eyear}.feather")
    this_mag_data = this_mag_data.set_index("Date_UTC", drop=True)
    this_mag_data.index = pd.to_datetime(this_mag_data.index)
    n_data, e_data = this_mag_data["dbn_geo"].rename(f"{station_name}_dbn"), \
        this_mag_data["dbe_geo"].rename(f"{station_name}_dbe")

    return n_data, e_data


def create_delays(df, name, time=20):
    for delay in np.arange(1, int(time) + 1):
        df[name + '_%s' % delay] = df[name].shift(delay).astype('float32')

def split_storm_list(storm_list, proportions=[0.7, 0.15, 0.15]):
    assert sum(proportions) == 1, "STORM SPLIT ERROR: The sum of the elements of 'proportions' must equal 1"
    assert sum([p < 0 for p in proportions]) == 0, "STORM SPLIT ERROR: No element of 'proportions' can be negative"

    num_train_storms = int(proportions[0] * len(storm_list))  # The int() function rounds down; e.g. int(4.9) == 4
    num_validation_storms = int(proportions[1] * len(storm_list))
    # num_test_storms = int(1. - num_train_storms - num_validation_storms)  # Just for your reference

    # Save a random subset of storms as the training list, then remove them from the storm list
    rng = np.random.default_rng()
    to_drop = rng.choice(len(storm_list), size=num_train_storms, replace=False)  # Must sample without replacement
    train_storm_list = storm_list.iloc[to_drop]
    storm_list = storm_list.drop(to_drop, axis="index").reset_index()

    # Save a random subset of storms as the validation list, then remove them from the storm list
    to_drop = rng.choice(len(storm_list), size=num_validation_storms, replace=False)  # Must sample without replacement
    valid_storm_list = storm_list.iloc[to_drop]
    test_storm_list = storm_list.drop(to_drop, axis="index").reset_index()  # The remaining storm list is now identical to the test list

    return train_storm_list, valid_storm_list, test_storm_list


def get_storm_data(all_data, storms_sublist, lead=12, recovery=24, status=""):
    storm_data = []  # Will be a list of dataframes, each one corresponding to one storm
    for date in tqdm.tqdm(storms_sublist["dates"], desc=f"Selecting {status} storms"):
        stime = (dt.datetime.strptime(date, '%m-%d-%Y %H:%M')) - pd.Timedelta(hours=lead)  # Storm onset time
        etime = (dt.datetime.strptime(date, '%m-%d-%Y %H:%M')) + pd.Timedelta(hours=recovery)  # Storm end time
        this_storm = all_data[(all_data.index >= stime) & (all_data.index <= etime)]
        this_storm.dropna(inplace=True)
        if len(this_storm) > 0:
            storm_data.append(this_storm)

    return storm_data


def preprocess_data(syear, eyear, stations_list, station_coords_list, sec_coords_list, omni_data_path,
                    supermag_data_path, iono_data_path, calculate_sec=True, proportions=[0.7, 0.15, 0.15 ], lead=12,
                    recovery=24):

    print("Loading OMNI data...")
    omni_data = load_omni(syear, eyear, data_path=omni_data_path)
    n_data, e_data = pd.DataFrame([]), pd.DataFrame([])
    print("Loading SuperMAG data...\n")
    for station_name in tqdm.tqdm(stations_list):
        this_n_data, this_e_data = load_supermag(station_name, syear, eyear, data_path=supermag_data_path)
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

        # Create an all-data dataframe so that NaNs and storms can be dropped while keeping indices aligned
        all_data = pd.concat([Z_matrix, omni_data], axis=1)

        print(f"Dropping NaNs from temporarily-combined SuperMAG and OMNI data.\nLength before dropping: "
              f"{len(all_data)}")
        all_data.dropna(axis=0, inplace=True)
        print(f"Length after dropping: {len(all_data)}\n")

        # Split storm list into training and test storms
        storm_list = pd.read_csv("stormList.csv", header=None, names=["dates"])
        train_storm_list, valid_storm_list, test_storm_list = split_storm_list(storm_list, proportions=proportions)

        # Keep only storm-time training data
        all_storm_data = []  # Its elements are three lists of dataframes, one for each split of the dataset
        for sublist, status in \
                zip([train_storm_list, valid_storm_list, test_storm_list], ["training", "validation", "test"]):
            subdata = get_storm_data(all_data, sublist, lead, recovery, status)  # List of storm-time dataframes
            all_storm_data.append(subdata)

        del all_data

        # Separate back out OMNI and SuperMAG data from each combined DataFrame
        omni_params = ["B_Total", "BX_GSE", "BY_GSM", "BZ_GSM", "flow_speed",
                       "Vx", "Vy", "Vz", "proton_density", "T", "Pressure", "E_Field"]
        split_sec_data, split_omni_data = [], []  # Each is a list of lists of storm-time dataframes
        for dataset, status in zip(all_storm_data, ["training", "validation", "test"]):
            print(f"Generating SEC coefficients for {status} set. "
                  "To load them from a file instead, use 'calculate_sec=False'")
            this_sec_dataset, this_omni_dataset = [], []  # Each is a list of storm-time dataframes for a certain split
            storm_num = 0
            for storm_df in tqdm.tqdm(dataset):
                this_omni_data = storm_df[omni_params]
                this_omni_dataset.append(this_omni_data)
                this_Z_matrix = storm_df.drop(columns=omni_params)

                this_sec_data = gen_current_data(this_Z_matrix, station_coords_list, sec_coords_list, epsilon=1e-3,
                                                 disable_tqdm=True)
                reduced_index = this_Z_matrix.index
                this_sec_data.index = reduced_index
                this_sec_data.columns = this_sec_data.columns.astype(str)  # Column names must be strings to save to feather
                this_sec_dataset.append(this_sec_data)
                this_sec_data.reset_index().to_feather(f"{iono_data_path}storms/I_{syear}-{eyear}_{status}-{storm_num}.feather")
                split_sec_data.append(this_sec_dataset)
                storm_num += 1
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

    # split_omni_data and split_sec_data are lists which each have three elements:
    # the training data, the validation data, and the test data
    return split_omni_data, split_sec_data, test_n_data, test_e_data
