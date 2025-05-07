import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# function to downsample signal based on bonsai sampling rate
def downsample_pp_data(pp_df, position_df):
    avg_sampling_rate_pp = float(1 / pp_df['time_seconds'].diff().mean())  # 130
    avg_sampling_rate_bonsai = float(1 / position_df['time_seconds'].diff().mean())  # 30
    length = int(len(pp_df['time_seconds']) / (avg_sampling_rate_pp/avg_sampling_rate_bonsai))
    indices = (np.arange(length) * (avg_sampling_rate_pp/avg_sampling_rate_bonsai)).astype(int)
    pp_df_downsampled = pp_df.iloc[indices]  # use indices to extract rows of df

    return pp_df_downsampled


# function to reduce noise in sync pulse data
def reduce_noise(pulses, threshold, high_level=5):
    """
    Clean up the signal by assigning value lower than the threshold to 0
    and those higher than the threshold the high_level. The high_level is set to 5 by default
    to match with the oe signal. Setting the high_level is necessary because signal drift in the bonsai high level
    may lead to uneven weighting of the value in the correlation calculation
    """
    pulses[pulses < threshold] = 0
    pulses[pulses >= threshold] = high_level
    return pulses


# function to pad shorter array with 0s
def pad_shorter_array_with_0s(array1, array2):
    if len(array1) < len(array2):
        array1 = np.pad(array1, (0, len(array2)-len(array1)), 'constant')
    if len(array2) < len(array1):
        array2 = np.pad(array2, (0, len(array1)-len(array2)), 'constant')
    return array1, array2


def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index


# function to remove extra pulses that one dataset has but not the other
def trim_arrays_find_starts(pp_df, position_df, bonsai_sampling_rate):
    pp_time = pp_df.time_seconds
    bonsai_time = position_df.synced_time_estimate
    pp_start_index = 19 * int(bonsai_sampling_rate)  # bonsai sampling rate times 19 seconds
    pp_start_time = pp_time.values[19 * int(bonsai_sampling_rate)]
    bonsai_start_index = find_nearest(bonsai_time.values, pp_start_time)
    return pp_start_index, bonsai_start_index


# function to find rising edge of pulse to be synced
def detect_last_zero(signal):
    """
    signal is a already thresholded binary signal with 0 and 1
    return the index of the last 0 before the first 1
    """
    # index of first zero value + index of first non-zero value
    if signal[0] != 0:
        signal[0:10] = 0
    first_nonzero_index = (np.argmin(signal)) + (np.nonzero(signal)[0][0])  # bug here if first_index_in_signal not 0
    assert first_nonzero_index == np.nonzero(signal)[0][0], 'Error, sync signal does not start at zero'
    last_zero_index = first_nonzero_index - 1
    return last_zero_index


def make_sync_plot(path, data, name, plot_color='black'):
    save_path = path + '/sync_pulses/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.figure()
    plt.plot(data, color=plot_color, label=name)
    plt.legend()
    plt.savefig(save_path + "/" + name + '_sync_pulses.png')
    plt.close()


# function to merge two dataframes of different lengths
def merge_two_dataframes(df1, df2):
    length_diff = np.abs(len(df1) - len(df2))  # cut off end of longer df to make dfs the same length

    if len(df1) > len(df2):
        df1 = df1.iloc[:-length_diff]

    elif len(df1) < len(df2):
        df2 = df2.iloc[:-length_diff]

    # reset indices & merge
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    merged_df = pd.concat([df1, df2], axis=1)

    return merged_df


def merge_synchronised_data(pp_df, position_df):
    """
    Merges dataframes containing position and signal
    Handles dataframes of different lengths by cutting off data from before the lag and removing excess rows from
    larger dataframe. Returns merged data frame containing position and calcium signal data
    """

    # slice data based on the first synced timestamp
    pp_time = pp_df.time_seconds.values
    bonsai_time = position_df.synced_time_bonsai.values

    if bonsai_time[0] > pp_time[0]:
        pp_sync_index = find_nearest(pp_time, bonsai_time[0])
        pp_df = pp_df.iloc[pp_sync_index::, :]

    elif bonsai_time[0] < pp_time[0]:
        bonsai_sync_index = find_nearest(bonsai_time, pp_time[0])
        position_df = position_df.iloc[bonsai_sync_index::, :]

    merged_df = merge_two_dataframes(position_df, pp_df)
    merged_df = merged_df.drop(['synced_time_bonsai'], axis=1)

    print('Signal and animal position were synchronised successfully.')
    return merged_df


def synchronise_data(pp_df, position_df, path):
    """
    Adapted from code developed for synchronising open ephys & bonsai (developed by Klara Gerlei)

    The pyPhotometry data and positional data are synchronised based on sync pulses sent both to the open ephys and
    bonsai systems. The pyPhotometry system receives pulses as a digital input (0 when no pulse, 1 when pulse) and
    Bonsai detects intensity from an LED that lights up whenever the pulse is sent to pyPhotometry. The pulses have
    20-60 second randomised gaps between them. The recordings do not necessarily start at the same time.
    PyPhotometry samples at 130 Hz and Bonsai at 30 Hz, but the webcam frame rate is not precise.

    1) PyPhotometry signal is downsampled to match the sampling rate of Bonsai, calculated based on the average interval
    between time stamps.
    2) Correlations are calculated between pulse data for pyphotometry & bonsai
    3) Lag is estimated based on the highest correlation and Bonsai is shifted by this lag
    2) 19s of data is removed to ensure corresponding pulse for fine lag calculation
    3) Lag is calculated from shifted time stamps using the rising edges of the first sync pulses
    4) Bonsai is shifted again by this lag
    5) Shifted bonsai data and downsampled pyPhotometry data are returned for merging in subsequent function.

    After these steps there will still be a small lag of ~ 30 ms (around the frame rate of the camera)
    """

    print('----------------------------------------------------------------------------------------')
    print('Synchronising photometry data with positional data...')
    print('----------------------------------------------------------------------------------------')

    # downsample pyPhotometry data
    pp_df = downsample_pp_data(pp_df, position_df)
    pp_pulses = pp_df['sync_pulse'].values
    bonsai_pulses = position_df['syncLED'].values

    # remove noise from pulse data & pad shorter array with 0s
    LED_threshold = np.median(position_df.syncLED) + (np.std(position_df.syncLED) * 6)  # 6 SDs > LED median intensity
    bonsai_pulses = reduce_noise(bonsai_pulses, LED_threshold)
    pp_pulses = reduce_noise(pp_pulses, 1)
    bonsai_pulses, pp_pulses = pad_shorter_array_with_0s(bonsai_pulses, pp_pulses)

    make_sync_plot(path, bonsai_pulses, 'bonsai')
    make_sync_plot(path, pp_pulses, 'digital', plot_color='red')

    # generate correlation array for bonsai & pyphotometry sync pulses
    corr = np.correlate(bonsai_pulses, pp_pulses, "full")

    # calculate lag based on max correlations between pulse data
    avg_sampling_rate_bonsai = float(1 / position_df['time_seconds'].diff().mean())  # 30
    lag = (np.argmax(corr) - (corr.size + 1) / 2) / avg_sampling_rate_bonsai
    print(f'Lag before synchronisation is {lag} seconds')
    position_df['synced_time_estimate'] = position_df.time_seconds - lag

    # remove first 19 seconds ensure a corresponding pulse
    pp_start, bonsai_start = trim_arrays_find_starts(pp_df, position_df, avg_sampling_rate_bonsai)
    trimmed_pp_time = pp_df.time_seconds.values[pp_start:]
    trimmed_pp_pulses = pp_pulses[pp_start:len(trimmed_pp_time)]
    trimmed_bonsai_time = position_df['synced_time_estimate'].values[bonsai_start:]
    trimmed_bonsai_pulses = bonsai_pulses[bonsai_start:]

    # find index of first pulse to calculate residual lag
    pp_rising_edge_index = detect_last_zero(trimmed_pp_pulses)
    pp_rising_edge_time = trimmed_pp_time[pp_rising_edge_index]
    bonsai_rising_edge_index = detect_last_zero(trimmed_bonsai_pulses)
    bonsai_rising_edge_time = trimmed_bonsai_time[bonsai_rising_edge_index]
    lag2 = pp_rising_edge_time - bonsai_rising_edge_time
    lag_to_return = lag - lag2

    # print lag information
    if abs(lag2) < 2.5:
        print(f'After synchronisation, rising edge lag is {lag2} seconds')
        position_df['synced_time_bonsai'] = position_df.synced_time_estimate + lag2
        synced_position_df = position_df[['synced_time_bonsai', 'position_x', 'position_x_pixels', 'position_y',
                                          'position_y_pixels', 'hd', 'speed']].copy()
        pp_df2 = pp_df[['time_seconds', 'signal', 'isosbestic', 'z_score']]

    else:
        # lag unexpectedly large, bug in code
        print('Lag is:' + str(lag2))
        raise ValueError('Potential sync error.')

    return synced_position_df, pp_df2, lag_to_return
