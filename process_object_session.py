import numpy as np
import pandas as pd
import sys
import statistics as stat
import synchronise_data
import file_utility
import math
import plotting_utility
import os
pd.options.mode.chained_assignment = None


def process_timestamps(df, reference, lag):
    # convert time to seconds
    df[['hours', 'minutes', 'seconds']] = df['time'].str.split(':', n=2, expand=True)
    df['hours'] = df['hours'].astype(int)
    df['minutes'] = df['minutes'].astype(int)
    df['seconds'] = df['seconds'].astype(float)
    df['time_seconds'] = df['hours'] * 3600 + df['minutes'] * 60 + df['seconds']
    df['time_seconds'] = df['time_seconds'] - reference  # align to reference from position dataframe
    df['synced_time'] = df.time_seconds - lag  # synchronise with sync pulse using calculated lag
    df = df.drop(['time_seconds', 'hours', 'minutes', 'seconds', 'time'], axis=1)

    return df


def curate_object_data(df, reference, lag, xmin, ymin, pixel_ratio=440):
    """
    :param df: pd.dataframe
    :param reference: zero reference (sampling points) from position data file
    :param lag: calculated lag from position x photometry synchronisation
    :param xmin: min x value in position data
    :param ymin: min y value in position data
    :return: pd.dataframe with cols 'pos_x_cm', 'pos_y_cm', 'synced_time'
    """
    df = process_timestamps(df, reference, lag)
    df['x_pixels'] = df.x_pixels - xmin  # subtract to shift x to 0
    df['y_pixels'] = df.y_pixels - ymin  # subtract to shift y to 0
    df['pos_x_cm'] = df.x_pixels / pixel_ratio * 100  # convert to cm, 440 is pixel ratio
    df['pos_y_cm'] = df.y_pixels / pixel_ratio * 100

    return df


# Check if object was moved, and return start index for second object position
def check_for_object_vector_session(df):
    object_moved, change_index = False, None

    rolling_x = np.around(df.iloc[:, 3].rolling(window=5).mean().values)  # rolling average of object positions
    rolling_y = np.around(df.iloc[:, 4].rolling(window=5).mean().values)
    mode_x, mode_y = stat.mode(rolling_x), stat.mode(rolling_y)  # mode of object centroid over whole session
    mode_x2 = stat.mode(list(filter(mode_x.__ne__, rolling_x)))  # second mode of object centroid whole session
    mode_y2 = stat.mode(list(filter(mode_y.__ne__, rolling_y)))
    diff_x, diff_y = abs(mode_x - mode_x2), abs(mode_y - mode_y2)  # absolute difference, small if no movement

    if (diff_x > 5) & (diff_y > 5):  # position must change on x & y axis
        object_moved = True
        x_indices = [np.where(rolling_x == mode_x)[0][0], np.where(rolling_x == mode_x2)[0][0]]
        y_indices = [np.where(rolling_y == mode_y)[0][0], np.where(rolling_y == mode_y2)[0][0]]
        change_index = max(max(x_indices), max(y_indices))  # pick larger index as change start

    return object_moved, change_index


def label_object_session(df, object_df):
    """
    :param df: df with animal position and signal data
    :param object_df: df with object positions
    :return: df with new 'session' column. Labels include 'baseline', 'object', and 'baseline_2'
    """

    df['session'] = np.nan  # create new column in dataframe with NaNs
    position = object_df.x_pixels  # x position values - NaN when object not present
    object_times, times = object_df.synced_time.values, df.time_seconds.values  # timestamps from object and signal file
    object_moved, move_index = check_for_object_vector_session(object_df)
    start, end = object_times[position.first_valid_index()], object_times[position.last_valid_index()]  # first & last detected positions

    try:  # find closest timestamps in signal file
        start_index, end_index = synchronise_data.find_nearest(times, start), synchronise_data.find_nearest(times, end)

        # label sessions based on indices
        df['session'][0:start_index] = 'baseline'
        df['session'][end_index + 1::] = 'baseline_2'

        if object_moved:
            print("This is an object-vector session. I will analyse data for both positions.")
            end_pos1, start_pos2 = object_times[move_index-75], object_times[move_index]
            start_index_2 = synchronise_data.find_nearest(times, start_pos2)   # start of second position
            end_index_2 = synchronise_data.find_nearest(times, end_pos1)  # end of first position (inc. 5 sec buffer)

            # label sessions based on indices around object movement
            df['session'][start_index:end_index_2+1] = 'object'
            df['session'][start_index_2:end_index + 1] = 'object_pos2'

        else:  # label session based on presence of object without movement
            df['session'][start_index:end_index + 1] = 'object'

    except ValueError:  # bug: will throw an error if positions are all NaNs
        print('There are no position values in the object file, object analysis will not run.')
        sys.exit()

    return df, object_moved, move_index


# find average position(s) of one object in the arena in cm
def find_object_positions(df, object_vector, change_index):
    if object_vector:  # look for two different positions around index where the object was moved
        pos1_x, pos1_y = df[0:(change_index-75)].pos_x_cm.mean(), df[0:(change_index-75)].pos_y_cm.mean()  # first position
        pos2_x, pos2_y = df[change_index::].pos_x_cm.mean(), df[change_index::].pos_y_cm.mean()  # second position
        positions = [pos1_x, pos1_y, pos2_x, pos2_y]

    else:  # assume single position and average across the whole session
        pos_x, pos_y = df.pos_x_cm.mean(), df.pos_y_cm.mean()
        positions = [pos_x, pos_y]

    return positions


def process_object_positions(df, object_filenames, reference, lag, xmin, ymin):
    """
    1. Checks folder for .csv files with names containing 'object' - expects 1 file per object
    2. Reads in object positions, curates, and returns as x and y coordinates in cm
    3. Returns number of objects and dataframe with object position(s) as averages
    """
    object_positions = []

    for fn in object_filenames:
        object_df = file_utility.read_object_data(fn)  # read object data
        object_df = curate_object_data(object_df, reference, lag, xmin, ymin)  # zero, shift, and convert to cm
        df, object_moved, move_index = label_object_session(df, object_df)
        positions = find_object_positions(object_df, object_moved, move_index)
        object_positions.append(positions)

    return df, object_positions, object_moved


def save_object_positions(path, positions):
    file_path = os.path.join(path, 'obj_positions.txt')
    if not os.path.isfile(file_path):
        with open(file_path, 'w') as file:
            file.write(str(positions))


# add column to dataframe with timestamps for each bout starting from 0
def calculate_time_for_each_bout(df, n_bouts):
    if n_bouts > 0:
        timestamps = pd.Series(np.linspace(-2, 5, 211).round(2), dtype='float64')
        adjusted_times = pd.concat([timestamps] * n_bouts, ignore_index=True)
        df['adjusted_time'] = adjusted_times

    return df


# find indices of start/end of each bout. By default, requires >1s between bouts
def find_ons_and_offs(indices, between_bouts_s=1):
    ends = np.where(np.diff(indices) > between_bouts_s*30)[0]
    starts = [0]  # first start
    for end in ends:  # add start for each end
        starts.append(end + 1)
    starts.pop()  # remove false start at the bottom of list
    return starts, ends


# create dataframe containing 7 seconds around object exploration
def make_bout_df(df, indices, starts, ends):
    df['bout'] = np.nan
    bouts = pd.DataFrame()
    index_counter, bout_counter = 0, 0

    for start in starts:
        start_index, end_index = indices[start], indices[ends[index_counter]]
        index_counter += 1

        if start_index < 60:  # skip first bout if there isn't sufficient baseline data (2s, 60 sampling points)
            pass
        elif end_index - start_index < 30:  # skip bout if shorter than 0.5s (15 sampling points)
            pass
        else:
            bout_counter += 1
            df['bout'][start_index-60:start_index+151] = bout_counter
            bout_data = df[start_index-60:start_index+151]
            bout_data = bout_data[['time_seconds', 'z_score', 'bout']].copy()
            bouts = pd.concat([bouts, bout_data], ignore_index=True)

    bouts = calculate_time_for_each_bout(bouts, bout_counter)

    return bouts


def find_bouts(df, x, y, threshold_cm):
    lower_x, upper_x = (x - threshold_cm), (x + threshold_cm)  # upper and lower bounds of x position
    lower_y, upper_y = (y - threshold_cm), (y + threshold_cm)  # upper and lower bounds of y position
    indices = df.query('position_x>@lower_x & position_x<@upper_x & position_y>@lower_y & position_y<@upper_y').index.tolist()
    starts, ends = find_ons_and_offs(indices)
    bouts = make_bout_df(df, indices, starts, ends)

    return bouts


def get_trial_dataframes(df):
    baseline_2 = None
    second_baseline = False

    baseline = df.loc[df["session"] == 'baseline']
    object_pos1 = df.loc[df["session"] == 'object'].reset_index()  # first position or only position

    if len(df.loc[df["session"] == 'baseline_2']) > 0:  # check for second baseline
        baseline_2 = df.loc[df["session"] == 'baseline_2'].reset_index()
        second_baseline = True

    return baseline, object_pos1, baseline_2, second_baseline


def save_bout_data(path, count, baseline, object_pos1, baseline2, object_pos2=None, second_position=False):
    if second_position:
        pos = '2_'
    else:
        pos = ''
    baseline['session'] = pos + 'baseline'  # add session labels to df
    object_pos1['session'] = pos + 'object'

    if object_pos2 is not None:  # add data for second position
        object_pos2['session'] = pos + 'object_pos2'
        bout_df = pd.concat([baseline, object_pos1, object_pos2], ignore_index=True)

    else:
        bout_df = pd.concat([baseline, object_pos1], ignore_index=True)

    if baseline2 is not None:  # add data for second baseline
        baseline2['session'] = pos + 'baseline_2'
        bout_df = pd.concat([bout_df, baseline2], ignore_index=True)

    if second_position:
        bout_df.to_pickle(path + "/dataframes/2_bout_data_object" + str(count + 1) + ".pkl")
    else:
        bout_df.to_pickle(path + "/dataframes/bout_data_object" + str(count+1) + ".pkl")


def analyse_single_object_session(df, positions, path):
    multiple_objects = False
    baseline2_bouts = None
    baseline, object_df, baseline2, baseline2_exists = get_trial_dataframes(df)  # split data for analysis
    threshold = file_utility.check_for_object_metadata(path)  # distance where animal is < 3 cm from object edge

    if len(positions) > 1:
        multiple_objects = True
        bout_ymin, bout_ymax = plotting_utility.find_axis_lims_for_multiple_objects(positions, object_df, max(threshold))

    object_count = 0
    for position in positions:
        if multiple_objects:
            threshold_cm = threshold[object_count]
        else:
            threshold_cm = threshold
        object_x, object_y = position[0], position[1]

        # dataframes for signal during baseline & object sessions when animal is near the object location
        baseline_bouts = find_bouts(baseline, object_x, object_y, threshold_cm)  # empty location
        object_bouts = find_bouts(object_df, object_x, object_y, threshold_cm)  # object

        # find universal axis limits for openfield and bout plots based on data
        ymin, ymax = plotting_utility.find_axis_lims_for_binned_data(baseline, object_df)
        if not multiple_objects:
            bout_ymin, bout_ymax = plotting_utility.find_y_axis_limits(baseline_bouts, object_bouts)
        ylims = [ymin, ymax, bout_ymin, bout_ymax]  # array with axis limits for plots

        if baseline2_exists:  # bouts at empty location after object removal
            baseline2_bouts = find_bouts(baseline2, object_x, object_y, threshold_cm)
            b2_ymin, b2_ymax = plotting_utility.calculate_y_min_and_max(baseline2_bouts, 'z_score')
            ylims[2] = min(bout_ymin, b2_ymin)
            ylims[3] = max(bout_ymax, b2_ymax)

        save_bout_data(path, object_count, baseline_bouts, object_bouts, baseline2=baseline2_bouts)  # save df with bout data

        # make plots
        file_utility.create_folder(path, '/object_plots/')  # folder for plots
        plotting_utility.make_object_plots(baseline, baseline_bouts, path, 'baseline_' + str(object_count+1), ylims)  # baseline plots
        plotting_utility.make_object_plots(object_df, object_bouts, path, 'object_' + str(object_count+1), ylims, object_x=object_x, object_y=object_y)  # object plots

        if baseline2_exists:  # plots for second baseline
            plotting_utility.make_object_plots(baseline2, baseline2_bouts, path, 'second_baseline_' + str(object_count+1), ylims)

        object_count += 1

    plotting_utility.make_combined_session_figure(path + '/object_plots')


def analyse_object_vector_session(df, positions, path):
    baseline2_pos1, baseline2_pos2 = None, None
    baseline, object_df, baseline_2, baseline2_exists = get_trial_dataframes(df)  # split data for analysis
    object_df_2 = df.loc[df["session"] == 'object_pos2'].reset_index()  # second position
    threshold_cm = file_utility.check_for_object_metadata(path)  # distance where animal is < 3 cm from object edge

    for position in positions:
        x_pos1, y_pos1 = position[0], position[1]
        x_pos2, y_pos2 = position[2], position[3]

        # dataframes for signal during baseline & object sessions when animal is near the object location
        baseline_pos1 = find_bouts(baseline, x_pos1, y_pos1, threshold_cm)  # empty location - position 1
        object_pos1 = find_bouts(object_df, x_pos1, y_pos1, threshold_cm)  # object - position 1
        baseline_pos2 = find_bouts(baseline, x_pos2, y_pos2, threshold_cm)  # empty location - position 2
        object_pos2 = find_bouts(object_df_2, x_pos2, y_pos2, threshold_cm)  # object - position 2
        object_empty_pos2 = find_bouts(object_df, x_pos2, y_pos2, threshold_cm)  # first object trial - position 2
        object_empty_pos1 = find_bouts(object_df_2, x_pos1, y_pos1, threshold_cm)  # second object trial - position 1

        # find universal axis limits for openfield and bout plots based on data
        ymin, ymax = plotting_utility.find_axis_lims_for_binned_data(baseline, object_df)
        bout_ymin1, bout_ymax1 = plotting_utility.find_y_axis_limits(baseline_pos1, object_pos1)  # axis limits for pos1
        bout_ymin2, bout_ymax2 = plotting_utility.find_y_axis_limits(baseline_pos2, object_pos2)  # axis limits for pos2
        ylims = [ymin, ymax, min(bout_ymin1, bout_ymin2), max(bout_ymax1, bout_ymax2)]  # array for y axis limits

        if baseline2_exists:  # add same data for empty locations after object removal
            baseline2_pos1 = find_bouts(baseline_2, x_pos1, y_pos1, threshold_cm)
            baseline2_pos2 = find_bouts(baseline_2, x_pos2, y_pos2, threshold_cm)
            b2_ymin1, b2_ymax1 = plotting_utility.calculate_y_min_and_max(baseline_pos1, 'z_score')
            b2_ymin2, b2_ymax2 = plotting_utility.calculate_y_min_and_max(baseline_pos2, 'z_score')
            binned_baseline2 = plotting_utility.make_pivot(baseline_2)

            # adjust universal y min/ax values if needed
            ylims[0] = min(ymin, math.floor(min(binned_baseline2.min())))
            ylims[1] = max(ymin, math.floor(max(binned_baseline2.max())))
            ylims[2] = min(bout_ymin1, bout_ymin2, b2_ymin1, b2_ymin2)
            ylims[3] = max(bout_ymax1, bout_ymax2, b2_ymax1, b2_ymax2)

        # save df with bout data
        save_bout_data(path, 0, baseline_pos1, object_pos1, baseline2=baseline2_pos1, object_pos2=object_empty_pos1)
        save_bout_data(path, 0, baseline_pos2, object_empty_pos2, baseline2=baseline2_pos2, object_pos2=object_pos2, second_position=True)

        # make plots
        file_utility.create_folder(path, '/object_plots/')
        plotting_utility.make_object_plots(baseline, baseline_pos1, path, 'baseline_pos1', ylims)
        plotting_utility.make_object_plots(object_df, object_pos1, path, 'object_pos1',  ylims, object_x=x_pos1, object_y=y_pos1)
        plotting_utility.make_object_plots(baseline, baseline_pos2, path, 'baseline_pos2', ylims)
        plotting_utility.make_object_plots(object_df_2, object_pos2, path, 'object_pos2', ylims, object_x=x_pos2, object_y=y_pos2)
        plotting_utility.make_object_plots(object_df, object_empty_pos2, path, 'object_empty_pos2', ylims, object_x=x_pos1, object_y=y_pos1)
        plotting_utility.make_object_plots(object_df_2, object_empty_pos1, path, 'object_empty_pos1', ylims, object_x=x_pos2, object_y=y_pos2)

        if baseline2_exists:
            plotting_utility.make_object_plots(baseline_2, baseline2_pos1, path, 'second_baseline_pos1', ylims)
            plotting_utility.make_object_plots(baseline_2, baseline2_pos2, path, 'second_baseline_pos2', ylims)

    plotting_utility.make_combined_session_figure(path + '/object_plots')


def analyse_object_session(df, path, object_files, zero_ref, lag, xmin, ymin):
    df, positions, object_vector = process_object_positions(df, object_files, zero_ref, lag, xmin, ymin)
    save_object_positions(path, positions)

    if object_vector:
        print('I have detected a session with a moving object. Analysing now...')
        analyse_object_vector_session(df, positions, path)

    else:
        print('This is a session with stationary object(s). Analysing now...')
        analyse_single_object_session(df, positions, path)
