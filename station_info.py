import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm

syear = 2010
eyear = 2014
stations_list = ['YKC', 'CBB', 'BLC', 'SIT', 'BOU', 'FRN', 'VIC', 'NEW', 'OTT', 'FRD', 'GIM', 'FCC', 'FMC', 'FSP',
                 'SMI', 'ISL', 'PIN', 'RAL', 'INK', 'CMO', 'IQA', 'GHC', 'LET', 'T03', 'C06', 'C07', 'C10', 'C12',
                 'T16', 'T32', 'T33', 'T36', 'T42']

geolat_list, geolon_list = [], []
n_na = []
for station in tqdm.tqdm(stations_list, desc="Loading SuperMAG data"):
    station_data = pd.read_feather(
                f"/data/ramans_files/mag-feather/magData-{station}_{syear}-{eyear}.feather")
    if station == "SZC":
        print(station_data["dbn_geo"].isna().sum())
    #station_data.dropna(inplace=True)
    station_data.reset_index(inplace=True)
    this_geolat = station_data.GEOLAT[0]
    this_geolon = station_data.GEOLON[0]
    geolat_list.append(this_geolat)
    geolon_list.append(this_geolon)
    n_na.append(station_data["dbn_geo"].isna().sum())

print(geolat_list, '\n', geolon_list)
print(n_na)
