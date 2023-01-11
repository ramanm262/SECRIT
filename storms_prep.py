import glob
import datetime as dt
import tqdm
import pandas as pd
from preprocessing_fns import *


def storm_extract(df, storm_list, B_param, lead, recovery):

    """
    Pulling out storms using a defined list of datetime strings, adding a lead and recovery time to it and
    appending each storm to a list which will be later processed.
    Inputs:
    df: dataframe of OMNI and Supermag data with the test sets already removed.
    storm_list: datetime list of storms minimums as strings.
    lead: how much time in hours to add to the beginning of the storm.
    recovery: how much recovery time in hours to add to the end of the storm.
    """

    storms, y = [], []					# initializing the lists
    stime, etime = [], []						# will store the resulting time stamps here then append them to the storm time df
    for date in storm_list:					# will loop through the storm dates, create a datetime object for the lead and recovery time stamps and append those to different lists
        stime.append((dt.datetime.strptime(date, '%m-%d-%Y %H:%M'))-pd.Timedelta(hours=lead))
        etime.append((dt.datetime.strptime(date, '%m-%d-%Y %H:%M'))+pd.Timedelta(hours=recovery))
    # adds the time stamp lists to the storm_list dataframes
    storm_list['stime'] = stime
    storm_list['etime'] = etime
    for start, end in zip(storm_list['stime'], storm_list['etime']):
        storm = df[(df.index >= start) & (df.index <= end)]
        storm = storm.dropna()
        if len(storm) != 0:
            storms.append(storm)			# creates a list of smaller storm time dataframes
    for storm in storms:
        storm.reset_index(drop=True, inplace=True)		# resetting the storm index and simultaneously dropping the date so it doesn't get trained on
        y.append(storm[B_param].to_numpy())				# creating the target array
        storm.drop([B_param], axis=1, inplace=True)  	# removing the target variable from the storm data so we don't train on it

    return storms, y
