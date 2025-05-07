"""
Script for preprocessing photometry data, adapted from: https://github.com/katemartian/Photometry_data_processing

Pre-processing steps:
    1. Smooth using a windowed average
    2. Find the baseline using lambda (can adjust in code?)
    3. Remove the baseline
    4. Standardise signals
    5. Fit reference signal to calcium signal using linear regression
    6. Align reference to signal
    7. Calculate z-score dF/F

Returns dataframe containing processed data in the following columns:
    time_seconds
    sync_pulse (binary)
    signal (standardized signal)
    isosbestic (standardized reference)
    z_score(signal - reference)

Also returns dataframe containing raw/semi-processed data in the following columns:
    time_seconds
    sync_pulse (binary)
    isosbestic
    gcamp
    gcamp_smoothed (preprocessing step 1)
    gcamp_corrected (after preprocessing step 3)
    isosbestic_corrected (after preprocessing step 3)
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
import file_utility
import plotting_utility
from math_utility import airPLS, smooth_signal


# extract raw data from .ppd file
def import_raw_data(data_folder, data_file):
    data = file_utility.import_ppd(os.path.join(data_folder, data_file))  # import ppd file
    gcamp_raw = data['analog_1']
    isosbestic_raw = data['analog_2']
    time_seconds = data['time']/1000
    sampling_rate = data['sampling_rate']  # 130
    sync_pulse = data['digital_1']  # sync pulses
    # pulse_times_1 = data['pulse_times_1']/1000 --- uncomment to retrieve raw pulse times
    df = pd.DataFrame(list(zip(time_seconds, sync_pulse, isosbestic_raw, gcamp_raw)),
                      columns=["time_seconds", "sync_pulse", "isosbestic", "gcamp"])

    return df, time_seconds, sampling_rate, sync_pulse


# preprocessing script - Kate Martian
def get_zdFF(df, smooth_win=10, remove=200, lambd=5e4, porder=1, itermax=50):
    """
    Calculates z-score dF/F signal based on fiber photometry calcium-independent and calcium-dependent signals

    Input
        reference: calcium-independent signal (usually 405-420 nm excitation), 1D array
        signal: calcium-dependent signal (usually 465-490 nm excitation for
                     green fluorescent proteins, or ~560 nm for red), 1D array
        smooth_win: window for moving average smooth, integer
        remove: the beginning of the traces with a big slope one would like to remove, integer
        Inputs for airPLS:
        lambd: parameter that can be adjusted by user. The larger lambda is,
                the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
        itermax: maximum iteration times

    Output
        df that contains columns of the gcamp & isosbestic data at preprocessing stages
        zdFF - z-score dF/F, 1D numpy array
    """
    raw_reference = df['isosbestic'][0:]
    raw_signal = df['gcamp'][0:]

    # smooth signal
    reference = smooth_signal(raw_reference, smooth_win)
    signal = smooth_signal(raw_signal, smooth_win)
    df['gcamp_smoothed'] = signal  # add to df
    df['isosbestic_smoothed'] = reference  # add to df

    # Remove slope using airPLS algorithm
    r_base = airPLS(reference, lambda_=lambd, porder=porder, itermax=itermax)
    s_base = airPLS(signal, lambda_=lambd, porder=porder, itermax=itermax)

    # Remove baseline from recording
    reference = (reference - r_base)
    signal = (signal - s_base)
    df['isosbestic_corrected'] = reference  # add to df
    df['gcamp_corrected'] = signal

    # Standardize signals
    reference = (reference - np.median(reference)) / np.std(reference)
    signal = (signal - np.median(signal)) / np.std(signal)

    # Align reference signal to calcium signal using non-negative robust linear regression
    lin = Lasso(alpha=0.0001, precompute=True, max_iter=1000, positive=True, random_state=9999, selection='random')
    n = len(reference)
    lin.fit(reference.reshape(n, 1), signal.reshape(n, 1))
    reference = lin.predict(reference.reshape(n, 1)).reshape(n, )

    # z dFF
    zdFF = (signal - reference)

    return df, signal, reference, zdFF


def preprocessing_plots(raw_df, folder):
    print('I am plotting the signal at different preprocessing steps...')

    # create new folder to save plots
    file_utility.create_folder(folder, '/preprocessing_plots')
    save_path = folder + '/preprocessing_plots'

    # generate and save preprocessing plots to folder
    plotting_utility.plot_signal(raw_df, 'gcamp', 'Raw Signal (V)', save_path, isosbestic=True, ref_col='isosbestic')
    plotting_utility.plot_signal(raw_df, 'gcamp_smoothed', 'Smoothed Signal (V)', save_path, isosbestic=True, ref_col='isosbestic_smoothed')
    plotting_utility.plot_signal(raw_df, 'gcamp_corrected', 'Processed Signal (V)', save_path, isosbestic=True, ref_col='isosbestic_corrected')
    plotting_utility.plot_signal(raw_df, 'gcamp', 'Raw Signal (V)', save_path, isosbestic=True, ref_col='isosbestic', around_peak=True)
    plotting_utility.plot_signal(raw_df, 'gcamp_smoothed', 'Smoothed Signal (V)', save_path, isosbestic=True, ref_col='isosbestic_smoothed', around_peak=True)
    plotting_utility.plot_signal(raw_df, 'gcamp_corrected', 'Processed Signal (V)', save_path, isosbestic=True, ref_col='isosbestic_corrected', around_peak=True)

    # make combined figure
    plotting_utility.make_combined_preprocessing_figure(save_path, 'combined_preprocessing_plot')


def preprocessing(folder, data_file):
    print('----------------------------------------------------------------------------------------')
    print('Preprocessing raw signal...')
    print('----------------------------------------------------------------------------------------')

    raw_df, time_seconds, sampling_rate, sync_pulse = import_raw_data(folder, data_file)  # import raw data
    raw_df, signal, reference, z_dff = get_zdFF(raw_df)  # process data

    # compile lists into dataframe
    print("I am creating a dataframe that contains the processed data...")
    df = pd.DataFrame(list(zip(time_seconds, sync_pulse, signal, reference, z_dff)),
                      columns=["time_seconds", "sync_pulse", "signal", "isosbestic", "z_score"])

    # generate plots of preprocessing steps
    preprocessing_plots(raw_df, folder)
    print("The data for", folder, "was processed successfully.")

    return raw_df, df
