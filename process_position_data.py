"""
Adapted from open_field_spatial_data.py (developed by Klara Gerlei & Teris Tam)

Bonsai file is a csv that contains the following information in each line:
    date of recording
    'T'
    exact time of given line
    x position of left side bead on headstage
    y position of left side bead on headstage
    x position of right side bead on headstage
    y position of right side bead on headstage
    intensity of sync LED

example line: 2018-03-06T15:34:39.8242304+00:00 106.0175 134.4123 114.1396 148.1054 1713

Returns pandas dataframe with data arranged in the following columns:
    time_seconds
    position_x
    position_x_pixels
    position_y
    position_y_pixels
    hd
    syncLED (LED intensity)
    speed
"""
import pandas as pd
import numpy as np
import math_utility
from scipy.interpolate import interp1d


def convert_time_to_seconds(position_data):
    position_data[['hours', 'minutes', 'seconds']] = position_data['time'].str.split(':', n=2, expand=True)
    position_data['hours'] = position_data['hours'].astype(int)
    position_data['minutes'] = position_data['minutes'].astype(int)
    position_data['seconds'] = position_data['seconds'].astype(float)
    position_data['time_seconds'] = position_data['hours']*3600 + position_data['minutes']*60 + position_data['seconds']
    zero_reference = position_data['time_seconds'][0]
    position_data['time_seconds'] = position_data['time_seconds'] - position_data['time_seconds'][0]

    return position_data, zero_reference


def read_position(path_to_bonsai_file):
    is_found = False
    position_data = None
    zero_reference = None

    try:
        position_data = pd.read_csv(path_to_bonsai_file, sep=' ', header=None)
        position_data = position_data.iloc[:, :-1]  # remove column of NaNs
        position_data[['date', 'time']] = position_data[0].str.split('T', n=1, expand=True)
        position_data[['time', 'str_to_remove']] = position_data['time'].str.split('+', n=1, expand=True)
        position_data = position_data.drop([0, 'str_to_remove'], axis=1)  # remove column that was split into date/time
        position_data.columns = ['x_left', 'y_left', 'x_right', 'y_right', 'syncLED', 'date', 'time']
        position_data, zero_reference = convert_time_to_seconds(position_data)
        is_found = True

    except FileNotFoundError:
        print('There is no position data for this recording. I will process signal without animal position.')

    return is_found, position_data, zero_reference


def resample_data(pos, fs):
    """
    Resample data so that they are of exact sampling rate because sometimes the FPS of the camera is not stable.
    Assume df has a time_seconds column
    """
    t = pos.time_seconds.values
    t2 = np.arange(0, t[-1], 1/fs)  # set end to t[-1] to avoid extrapolation, which may lead to error
    df = {}
    for col in pos.columns:
        f = interp1d(t, pos[col].values)
        df[col] = f(t2)

    df['time_seconds'] = t2
    df2return = pd.DataFrame(df)

    return df2return


def calculate_speed(position_data):
    elapsed_time = position_data['time_seconds'].diff()
    distance_travelled = np.sqrt(position_data['x_left'].diff().pow(2) + position_data['y_left'].diff().pow(2))
    position_data['speed_left'] = distance_travelled / elapsed_time
    distance_travelled = np.sqrt(position_data['x_right'].diff().pow(2) + position_data['y_right'].diff().pow(2))
    position_data['speed_right'] = distance_travelled / elapsed_time
    return position_data


def remove_jumps(position_data, pixel_ratio):
    max_speed = 1  # m/s, anything above this is not realistic
    max_speed_pixels = max_speed * pixel_ratio
    speed_exceeded_left = position_data['speed_left'] > max_speed_pixels
    position_data['x_left_without_jumps'] = position_data.x_left[speed_exceeded_left == False]
    position_data['y_left_without_jumps'] = position_data.y_left[speed_exceeded_left == False]

    speed_exceeded_right = position_data['speed_right'] > max_speed_pixels
    position_data['x_right_without_jumps'] = position_data.x_right[speed_exceeded_right == False]
    position_data['y_right_without_jumps'] = position_data.y_right[speed_exceeded_right == False]

    remains_left = (len(position_data) - speed_exceeded_left.sum())/len(position_data)*100
    remains_right = (len(position_data) - speed_exceeded_right.sum())/len(position_data)*100
    print('{} % of right side tracking data, and {} % of left side'
          ' remains after removing the ones exceeding speed limit.'.format(remains_right, remains_left))
    return position_data


def get_distance_of_beads(position_data):
    distance_between_beads = np.sqrt((position_data['x_left'] - position_data['x_right']).pow(2) +
                                     (position_data['y_left'] - position_data['y_right']).pow(2))
    return distance_between_beads


def remove_points_where_beads_are_far_apart(position_data):
    minimum_distance = 40
    distance_between_beads = get_distance_of_beads(position_data)
    distance_exceeded = distance_between_beads > minimum_distance
    position_data['x_left_cleaned'] = position_data.x_left_without_jumps[distance_exceeded == False]
    position_data['x_right_cleaned'] = position_data.x_right_without_jumps[distance_exceeded == False]
    position_data['y_left_cleaned'] = position_data.y_left_without_jumps[distance_exceeded == False]
    position_data['y_right_cleaned'] = position_data.y_right_without_jumps[distance_exceeded == False]
    return position_data


def curate_position(position_data, pixel_ratio):
    position_data = remove_jumps(position_data, pixel_ratio)
    position_data = remove_points_where_beads_are_far_apart(position_data)
    return position_data


def calculate_position(position_data):
    position_data['position_x_tmp'] = (position_data['x_left_cleaned'] + position_data['x_right_cleaned']) / 2
    position_data['position_y_tmp'] = (position_data['y_left_cleaned'] + position_data['y_right_cleaned']) / 2

    position_data['position_x'] = position_data['position_x_tmp'].interpolate()  # interpolate missing data
    position_data['position_y'] = position_data['position_y_tmp'].interpolate()
    return position_data


def calculate_head_direction(position):
    position['head_dir_tmp'] = np.degrees(np.arctan((position['y_left_cleaned'] + position['y_right_cleaned']) /
                                                    (position['x_left_cleaned'] + position['x_right_cleaned'])))
    rho, hd = math_utility.cart2pol(position['x_right_cleaned'] - position['x_left_cleaned'],
                                    position['y_right_cleaned'] - position['y_left_cleaned'])
    position['hd'] = np.degrees(hd)
    position['hd'] = position['hd'].interpolate()  # interpolate missing data
    return position


def shift_to_start_from_zero_at_bottom_left(position_data):
    # this is copied from MATLAB script, 0.0001 is here to 'avoid bin zero in first point'
    xmin = min(position_data.position_x[~np.isnan(position_data.position_x)])
    ymin = min(position_data.position_y[~np.isnan(position_data.position_y)])
    position_data['position_x'] = position_data.position_x - xmin
    position_data['position_y'] = position_data.position_y - ymin

    return position_data, xmin, ymin


def convert_to_cm(position_data, pixel_ratio):
    position_data['position_x_pixels'] = position_data.position_x
    position_data['position_y_pixels'] = position_data.position_y
    position_data['position_x'] = position_data.position_x / pixel_ratio * 100
    position_data['position_y'] = position_data.position_y / pixel_ratio * 100
    return position_data


def calculate_central_speed(position_data):
    elapsed_time = position_data['time_seconds'].diff()
    distance_travelled = np.sqrt(position_data['position_x'].diff().pow(2) + position_data['position_y'].diff().pow(2))
    position_data['speed'] = distance_travelled / elapsed_time
    return position_data


# read position data from Bonsai
def analyse_position(position_file, pixel_ratio=440, do_resample=True):
    print('----------------------------------------------------------------------------------------')
    print('Processing positional information...')
    print('----------------------------------------------------------------------------------------')
    print("Importing Bonsai file...")
    is_found, position_data, zero_reference = read_position(position_file)  # raw position data from bonsai output
    xmin = None
    ymin = None

    if is_found:
        if do_resample:
            position_data = position_data.drop(['date', 'time', 'hours', 'minutes', 'seconds'], axis=1)
            position_data = resample_data(position_data, 30)

        position_data = calculate_speed(position_data)
        position_data = curate_position(position_data, pixel_ratio)  # remove jumps from data & when beads are far apart
        position_data = calculate_position(position_data)  # get central position and interpolate missing data
        position_data = calculate_head_direction(position_data)  # use coord from two beads to get hd and interpolate
        position_data, xmin, ymin = shift_to_start_from_zero_at_bottom_left(position_data)
        position_data = convert_to_cm(position_data, pixel_ratio)
        position_data = calculate_central_speed(position_data)
        position_data = position_data[['time_seconds', 'position_x', 'position_x_pixels', 'position_y',
                                       'position_y_pixels', 'hd', 'syncLED', 'speed']].copy()
        print("Position data has been processed successfully.")

    return is_found, position_data, zero_reference, xmin, ymin
