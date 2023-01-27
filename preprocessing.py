from sec import *


def load_omni(syear, eyear, data_path):
    omni_data = pd.read_feather(data_path + f"/omniData-{syear}-{eyear}-interp-None.feather")
    omni_data = omni_data.rename(columns={"Epoch": "Date_UTC"})
    omni_data.set_index("Date_UTC", inplace=True, drop=True)
    omni_data = omni_data[["B_Total", "BX_GSE", "BY_GSM", "BZ_GSM", "flow_speed",
                           "Vx", "Vy", "Vz", "proton_density", "T", "Pressure", "E_Field"]]  # Only use these features

    return omni_data


def load_supermag(station_name, syear, eyear, data_path):
    this_mag_data = pd.read_feather(data_path + f"magData-{station_name}_{syear}-"
                                    f"{eyear}.feather")
    this_mag_data = this_mag_data.set_index("Date_UTC", drop=True)
    this_mag_data.index = pd.to_datetime(this_mag_data.index)
    n_data, e_data = this_mag_data["dbn_geo"].rename(f"{station_name}_dbn"), \
        this_mag_data["dbe_geo"].rename(f"{station_name}_dbe")

    return n_data, e_data


def preprocess_data(syear, eyear, stations_list, station_coords_list, sec_coords_list, omni_data_path,
                    supermag_data_path, iono_data_path, calculate_sec=True):

    print("Loading OMNI data...")
    omni_data = load_omni(syear, eyear, data_path=omni_data_path)
    n_data, e_data = pd.DataFrame([]), pd.DataFrame([])
    print("Loading SuperMAG data...")
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

        print(f"Dropping NaNs from SuperMAG data.\nLength before dropping: {len(Z_matrix)}")
        Z_matrix.dropna(axis=0, inplace=True)
        print(f"Length after dropping: {len(Z_matrix)}")
        print("Generating SEC coefficients. If you want to load them from a file instead, use 'calculate_sec=False'")
        sec_data = gen_current_data(Z_matrix, station_coords_list, sec_coords_list, epsilon=1e-4)
        sec_data.index = Z_matrix.index
        sec_data.columns = sec_data.columns.astype(str)  # Column names must be strings in order to save to feather
        sec_data.reset_index().to_feather(iono_data_path + f"I_{syear}-{eyear}.feather")
    else:
        sec_data = pd.read_feather(iono_data_path + f"I_{syear}-{eyear}.feather")
        sec_data = sec_data.rename(columns={"index": "Date_UTC"})
        sec_data.set_index("Date_UTC", inplace=True, drop=True)

    print(omni_data.shape)
    print(len(sec_data.index))
    omni_data = omni_data.loc[sec_data.index]
    n_data = n_data.loc[sec_data.index]
    e_data = e_data.loc[sec_data.index]

    print(omni_data.shape)
    print(len(sec_data.index))

    return omni_data, n_data, e_data, sec_data

