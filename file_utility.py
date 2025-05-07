import glob
import sys
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import re
import os
import math


# ---------------------------------------------------------------------------------------------------------------------
# General utility and scripts for identifying and reading Bonsai and pyPhotometry files
# ---------------------------------------------------------------------------------------------------------------------
def create_folder(folder_path, new_folder):
    save_path = (folder_path + new_folder)

    try:
        os.mkdir(save_path)

    except OSError:
        pass

    return save_path


def get_file_paths(folder_path):
    """
    Expects last 10 characters of folder name to match the last 10 characters of the data files
    eg. 1396_OF_2022-03-22 (folder), 1396_OF_2022-03-22.csv (file)
    returns list of file_paths at the following indices: bonsai file [0], pyPhotometry file [1]
    """
    names_to_find = [folder_path[-10:] + '.csv', '.ppd']
    try:
        file_paths = [glob.glob(folder_path + "/*" + names_to_find[0]),  # bonsai file
                      glob.glob(folder_path + "/*" + names_to_find[1])]  # pyPhotometry file

    except FileNotFoundError:
        print('Position and/or photometry files were not found for this recording. Analysis will not proceed.')
        sys.exit()

    return file_paths


def read_bonsai_file(data_file):
    df = pd.read_csv(data_file, sep=' ', header=None)
    df = df.iloc[:, :-1]  # remove column of NaNs
    df[['date', 'time']] = df[(len(df.columns))-1].str.split('T', n=1, expand=True)
    df[['time', 'str_to_remove']] = df['time'].str.split('+', n=1, expand=True)
    df = df.drop([(len(df.columns))-4, 'str_to_remove'], axis=1)  # drop split cols
    return df


def read_recording_list(file):
    with open(file, 'r') as f:
        recordings = []
        lines = f.readlines()
        for line in lines:
            recordings.append(line.strip())

    return recordings


def get_figure_paths(folder_path, session):
    """
    Searches '/object_plots' subfolder for figures corresponding to session (baseline, object, second_baseline)
    Returns list of filepaths used to make a combined object figure
    """
    label = session + '.png'
    names_to_find = ['trajectory_' + label,
                     'arena_heatmap_' + label,
                     'bout_heatmap_' + label,
                     'smoothed_signal_' + label]
    try:
        file_paths = [glob.glob(folder_path + "/*" + names_to_find[0]),  # trajectory
                      glob.glob(folder_path + "/*" + names_to_find[1]),  # arena heatmap
                      glob.glob(folder_path + "/*" + names_to_find[2]),  # bout heatmap
                      glob.glob(folder_path + "/*" + names_to_find[3])]  # signal

    except FileNotFoundError:
        sys.exit()

    return file_paths


# ---------------------------------------------------------------------------------------------------------------------
# Function for opening pyPhotometry data files in Python
# Copyright (c) Thomas Akam 2018.  Licenced under the GNU General Public License v3.
# Retrieved from Github 24-02-22: https://github.com/ThomasAkam/data_import/
# ---------------------------------------------------------------------------------------------------------------------
def import_ppd(file_path, low_pass=20, high_pass=0.01):
    """Function to import pyPhotometry binary data files into Python. The high_pass
    and low_pass arguments determine the frequency in Hz of highpass and lowpass
    filtering applied to the filtered analog signals. To disable highpass or lowpass
    filtering set the respective argument to None.  Returns a dictionary with the
    following items:
        'subject_ID'    - Subject ID
        'date_time'     - Recording start date and time (ISO 8601 format string)
        'mode'          - Acquisition mode
        'sampling_rate' - Sampling rate (Hz)
        'LED_current'   - Current for LEDs 1 and 2 (mA)
        'version'       - Version number of pyPhotometry
        'analog_1'      - Raw analog signal 1 (volts)
        'analog_2'      - Raw analog signal 2 (volts)
        'analog_1_filt' - Filtered analog signal 1 (volts)
        'analog_2_filt' - Filtered analog signal 2 (volts)
        'digital_1'     - Digital signal 1
        'digital_2'     - Digital signal 2
        'pulse_times_1' - Times of rising edges on digital input 1 (ms).
        'pulse_times_2' - Times of rising edges on digital input 2 (ms).
        'time'          - Time of each sample relative to start of recording (ms)
    """
    with open(file_path, 'rb') as f:
        header_size = int.from_bytes(f.read(2), 'little')
        data_header = f.read(header_size)
        data = np.frombuffer(f.read(), dtype=np.dtype('<u2'))
    # Extract header information
    header_dict = json.loads(data_header)
    volts_per_division = header_dict['volts_per_division']
    sampling_rate = header_dict['sampling_rate']
    # Extract signals.
    analog = data >> 1                     # Analog signal is most significant 15 bits.
    digital = ((data & 1) == 1).astype(int)  # Digital signal is least significant bit.
    # Alternating samples are signals 1 and 2.
    analog_1 = analog[ ::2] * volts_per_division[0]
    analog_2 = analog[1::2] * volts_per_division[1]
    digital_1 = digital[ ::2]
    digital_2 = digital[1::2]
    time = np.arange(analog_1.shape[0])*1000/sampling_rate  # Time relative to start of recording (ms).
    # Filter signals with specified high and low pass frequencies (Hz).
    if low_pass and high_pass:
        b, a = butter(2, np.array([high_pass, low_pass])/(0.5*sampling_rate), 'bandpass')
    elif low_pass:
        b, a = butter(2, low_pass/(0.5*sampling_rate), 'low')
    elif high_pass:
        b, a = butter(2, high_pass/(0.5*sampling_rate), 'high')
    if low_pass or high_pass:
        analog_1_filt = filtfilt(b, a, analog_1)
        analog_2_filt = filtfilt(b, a, analog_2)
    else:
        analog_1_filt = analog_2_filt = None
    # Extract rising edges for digital inputs.
    pulse_times_1 = (1+np.where(np.diff(digital_1) == 1)[0])*1000/sampling_rate
    pulse_times_2 = (1+np.where(np.diff(digital_2) == 1)[0])*1000/sampling_rate
    # Return signals + header information as a dictionary.
    data_dict = {'analog_1'      : analog_1,
                 'analog_2'      : analog_2,
                 'analog_1_filt' : analog_1_filt,
                 'analog_2_filt' : analog_2_filt,
                 'digital_1'     : digital_1,
                 'digital_2'     : digital_2,
                 'pulse_times_1' : pulse_times_1,
                 'pulse_times_2' : pulse_times_2,
                 'time'          : time}
    data_dict.update(header_dict)
    return data_dict


# ---------------------------------------------------------------------------------------------------------------------
# Scripts for reading in object files and object metadata
# ---------------------------------------------------------------------------------------------------------------------
def check_for_object_files(folder):
    is_found = None
    object_files = glob.glob(folder + "/*.csv")  # list all .csv files
    object_files = list(filter(lambda x: 'object' in x, object_files))  # filter for names containing object

    if len(object_files) != 0:
        is_found = True
        if len(object_files) > 1:  # sort filenames by object number if there is more than 1
            pattern = r'(.*?)(\d*)$'
            object_files = sorted(object_files, key=lambda s: (re.findall(pattern, s)[0][0], int(re.findall(pattern, s)[0][1] or 0)))

    return is_found, object_files, len(object_files)


def read_object_data(object_filename):
    """
    :param object_filename: file path to .csv containing object data
    :return: pd.dataframe with cols 'pos_x_pixels', 'pos_y_pixels' and 'time'

    Bonsai file contains the following information in each line:
    x position of object
    y position of object
    date of recording
    'T'
    exact time of given line
    example line: 366.2 344.5 2018-03-06T15:34:39.8242304+00:00
    """
    df = pd.read_csv(object_filename, sep=' ', header=None)
    df = df.iloc[:, :-1]  # remove column of NaNs
    df[['date', 'time']] = df[(len(df.columns)) - 1].str.split('T', n=1, expand=True)
    df[['time', 'str_to_remove']] = df['time'].str.split('+', n=1, expand=True)
    df = df.drop([(len(df.columns)) - 4, 'date', 'str_to_remove'], axis=1)  # drop split cols
    df.columns = ['x_pixels', 'y_pixels', 'time']

    return df


def calculate_exploration_threshold(length, width, item, distance=3):
    # return value for x distance (default = 3 cm) from object edge based on size, assumes centroid is in the middle
    if length == width and item != 'lego':  # symmetrical object
        threshold = length/2 + distance
    else:  # asymmetrical objects,
        # threshold = (math.hypot(length/2, width/2)) + distance # find x distance from longest corner distance to centroid
        threshold = ((length/2 + distance) + (width/2 + distance))/2  # take a middleground

    return threshold


def find_threshold(object, object_sizes, default):
    if object in object_sizes:
        l, w = object_sizes[object][0], object_sizes[object][1]
        threshold = calculate_exploration_threshold(l, w, object)  # calculate exploration threshold
        print('The object was', object, 'so a threshold of', threshold, 'cm will be used for analysis.')

    else:
        print("Unknown object specified. The default threshold of", default, "cm will be used for bout analysis.")
        threshold = default

    return threshold


def check_for_object_metadata(path):
    """
    Looks for metadata.txt in the recording folder and returns a distance from the object edge based on the size(s) of
    the objects specified. If there are two objects, an average size is used for this calculation.
    """
    filepath = path + "/metadata.txt"
    default = 7  # default threshold if no object information was provided
    # dictionary of objects with length and width values in cm
    object_sizes = {'duck': [7.5, 6], 'shaker': [5, 5], 'truck': [9.5, 5], 'gupot': [7.5, 7.5], 'cow': [9.5, 4],
                    'lego': [6.0, 6.0], 'firstlego': [6, 3], 'burger': [8, 8], 'ball': [7, 7], 'eggcup': [4.7, 4.7],
                    'jar': [6.8, 6.8]}

    if os.path.isfile(filepath):
        objects = []
        with open(filepath) as f:
            lines = f.readlines()
        for line in lines:
            if 'item' in line:
                line = line.strip()
                if '1' in line or '2' in line:
                    object_num = line[4]
                    objects.append(line[6:] + '_' + object_num)
                else:
                    objects.append(line[5:])  # object string

        if len(objects) == 2:
            object1, object2 = objects[0][:-2], objects[1][:-2]
            threshold1, threshold2 = find_threshold(object1, object_sizes, default), find_threshold(object2, object_sizes, default)
            threshold = [threshold1, threshold2]

        elif len(objects) == 1:
            threshold = find_threshold(objects[0], object_sizes, default)

        return threshold

    else:
        print("No object metadata was found. The default threshold of", default, "cm will be used for bout analysis.")
        return default


# key for sorting subfolders into correct order for plotting
def custom_sort_key(path):
    session_label = path.split('/')[-2]  # extract second-to-last element (session label)
    if 'baseline' in session_label and '1' in session_label and 'second' not in session_label:
        return 0
    elif 'object' in session_label and '1' in session_label:
        return 1
    elif 'pos2' in session_label:
        return 2
    elif 'second_baseline' in session_label:
        return 3


def get_openfield_figure_paths(folder_path):
    """
    Searches '/object_plots' subfolders for figures corresponding to trajectory & arena heatmap plots for each session.
    Returns list of filepaths that are used to make a combined figure for quick comparisons across sessions.
    """
    subfolders, file_paths = [], []

    for path in glob.glob(f'{folder_path}/*/'):  # list subfolders
        # exclude repeats of baseline session when object moved, or second object when there are >1 objects
        if ('baseline' in path and path[-2] == '2') or ('pos' not in path and int(path[-2]) > 1):
            pass
        else:
            subfolders.append(path)

    if len(subfolders) > 1:  # get paths to figures from each subfolder
        subfolders_sorted = sorted(subfolders, key=custom_sort_key)  # sort into correct order
        for folder in subfolders_sorted:
            trajectory = glob.glob(folder + "trajectory_*.png")  # get path to trajectory plot
            heatmap = glob.glob(folder + "arena_heatmap_*.png")  # get path to arena heatmap
            file_paths.append([trajectory, heatmap])

    return file_paths