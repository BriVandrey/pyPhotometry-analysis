import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import file_utility
import process_object_session
import os
import pandas as pd
import numpy as np
import math


def make_sync_plot(path, data, name, plot_color='black'):
    save_path = path + '/sync_pulses/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.figure()
    plt.plot(data, color=plot_color, label=name)
    plt.legend()
    plt.savefig(save_path + "/" + name + '_sync_pulses.png')
    plt.close()


# find y axis limits based on data, options to pad with standard deviations and round
def calculate_y_min_and_max(df, col, sd=0, remove_outliers=True, round_up=True):
    """
    :param df: dataframe
    :param col: column that contains data
    :param sd: number of standard deviations to add to min/max
    :param round_up: set to True to round up to the nearest whole integer
    """
    y_min, y_max = 0, 0  # placeholder of 0 if there is no bouts for session
    if len(df) > 0:
        vals = df[col].values
        if remove_outliers:  # remove outliers above/below 5 SD - avoids washed out plots
            p50, p90 = np.median(vals), np.percentile(vals, 90)
            rSig = (p90-p50)/1.2815
            vals = vals[abs(vals - p50) < rSig * 5]

        y_min, y_max = vals.min(), vals.max()

        if sd > 0:  # add standard deviations
            y_min, y_max = y_min - (sd * vals.std()), y_max + (sd * vals.std())

        if round_up:  # round to nearest integer
            y_min, y_max = round(y_min, 1), round(y_max, 1)
         #   y_min, y_max = math.floor(y_min), math.ceil(y_max)

    return y_min, y_max


# return universal y axis limits
def find_y_axis_limits(baseline, object, object2=None, baseline2=None):
    object2_min, object2_max, b2_min, b2_max = 0, 0, 0, 0  # placeholders
    b_min, b_max = calculate_y_min_and_max(baseline, 'z_score')  # baseline
    object_min, object_max = calculate_y_min_and_max(object, 'z_score')  # object

    if object2 is not None:  # session with moved object
        object2_min, object2_max = calculate_y_min_and_max(object2, 'z_score')
    if baseline2 is not None:  # second baseline session
        b2_min, b2_max = calculate_y_min_and_max(baseline2, 'z_score')

    ymin, ymax = min(b_min, object_min, object2_min, b2_min), max(b_max, object_max, object2_max, b2_max)

    return ymin, ymax


def find_axis_lims_for_multiple_objects(positions, df, threshold_cm):
    bouts = pd.DataFrame()
    for position in positions:
        x, y = position[0], position[1]
        object_bouts = process_object_session.find_bouts(df, x, y, threshold_cm)  # object
        bouts = pd.concat([bouts, object_bouts])
    ymin, ymax = calculate_y_min_and_max(bouts, 'z_score')  # axis limits for bouts

    return ymin, ymax


def find_axis_lims_for_binned_data(df, df2):
    binned_df, binned_df2 = make_pivot(df), make_pivot(df2),
    ymin = min(math.floor(min(binned_df.min())), math.floor(min(binned_df2.min())))
    ymax = max(math.ceil(max(binned_df.max())), math.ceil(max(binned_df2.max())))
    return ymin, ymax


# plot dF/F against time
def plot_signal(df, col, y_label, save_path, isosbestic=False, ref_col=None, around_peak=False):
    """
    Assumes dataframe has time in column 'time_seconds'
    If around_peak = True, signal is plotted over 10 s around peak amplitude
    """
    y_min, y_max = calculate_y_min_and_max(df, col, 2)  # find y axis scale
    plt.figure(figsize=(8, 4))
    ax = sns.lineplot(data=df, x='time_seconds', y=col)

    if isosbestic:
        y_min_2, y_max_2 = calculate_y_min_and_max(df, ref_col, 2)  # find y axis scale for isosbestic
        y_min = min(y_min, y_min_2)  # choose value to accommodate both signals
        y_max = max(y_max, y_max_2)
        ax = sns.lineplot(data=df, x='time_seconds', y=ref_col)
        ax.legend(['GCaMP', 'Isosbestic'], loc='upper right')

    ax.set_ylim(y_min, y_max)
    label = "whole_session"

    if around_peak:  # plot around peak DF/F value
        label = 'around_peak'
        peak_index = df[col].idxmax()  # find index of max DF/F value
        time_at_peak = df['time_seconds'][peak_index]
        ax.set_xlim(time_at_peak - 5, time_at_peak + 5)  # set x axis limits

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=21, labelpad=20)
    ax.set_ylabel(y_label, fontsize=21, labelpad=20)
    plt.tight_layout()
    plt.savefig(save_path + '/' + col + '_' + label + '.png', dpi=300)
    plt.close()


# add previously generated figure(s) to specified grid
def plot_to_grid(row, col, grid, path):
    """
    :param row: number of rows
    :param col: number of columns
    """
    image = mpimg.imread(path)
    image_plot = plt.subplot(grid[row, col])
    image_plot.axis('off')
    image_plot.imshow(image)


# make combined figure showing signal at different preprocessing steps
def make_combined_preprocessing_figure(save_path, plot_title):
    plt.figure(figsize=(20, 18))
    grid = plt.GridSpec(3, 2)
    plot_to_grid(0, 0, grid, save_path + '/gcamp_whole_session.png')  # raw data
    plot_to_grid(1, 0, grid, save_path + '/gcamp_smoothed_whole_session.png')  # smoothed
    plot_to_grid(2, 0, grid, save_path + '/gcamp_corrected_whole_session.png')  # corrected
    plot_to_grid(0, 1, grid, save_path + '/gcamp_around_peak.png')  # raw data around peak
    plot_to_grid(1, 1, grid, save_path + '/gcamp_smoothed_around_peak.png')  # smoothed
    plot_to_grid(2, 1, grid, save_path + '/gcamp_corrected_around_peak.png')  # corrected
    plt.tight_layout()
    plt.savefig(save_path + '/' + plot_title + '.png', dpi=300)
    plt.close()


# plot animal trajectory
def plot_trajectory(df, save_path, object_x=None, object_y=None, filename=False):
    """
    Assumes dataframe has columns 'position_x' & 'position_y'
    '"""
    plt.figure(figsize=(7, 6))
    plt.plot(df['position_x'], df['position_y'], color='black', linewidth=2, zorder=1, alpha=0.7)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("x (cm)", fontsize=25, labelpad=15)
    plt.ylabel("y (cm)", fontsize=25, labelpad=15)
    plt.title('Trajectory', fontsize=24)

    if object_x is not None:
        plt.scatter(object_x, object_y, s=1000, color='firebrick')
        plt.title('Trajectory with object location(s)', fontsize=26)

    plt.tight_layout()

    if not filename:
        plt.savefig(save_path + '/' + 'trajectory.png', dpi=300)
        plt.savefig(save_path + '/' + 'trajectory.eps', dpi=300)
    else:
        plt.savefig(save_path + '/' + 'trajectory_' + filename + '.png', dpi=300)
        plt.savefig(save_path + '/' + 'trajectory_' + filename + '.eps', dpi=300)
    plt.close()


# produce figure showing trajectory, signal over time, and signal 10 s around peak signal as a 1x3 grid
def make_combined_openfield_figure(path, plot_title, position=True):
    plt.figure(figsize=(10, 18))
    grid = plt.GridSpec(3, 1)

    if position:  # only plot trajectory if position is found
        plot_to_grid(0, 0, grid,  path + '/trajectory.png')

    plot_to_grid(1, 0, grid, path + '/z_score_whole_session.png')
    plot_to_grid(2, 0, grid, path + '/z_score_around_peak.png')
    plt.tight_layout()
    plt.savefig(path + '/' + plot_title + '.png', dpi=300)
    plt.close()


# plot data for open-field session
def plot_open_field_session(folder_path, df):
    print("I am now plotting signal in the openfield...")
    file_utility.create_folder(folder_path, "/openfield_plots")  # create folder for saving plots
    save_path = folder_path + "/openfield_plots"
    plot_trajectory(df, save_path)
    plot_signal(df, 'z_score', "Z-DF/F", save_path)
    plot_signal(df, 'z_score', "Z-DF/F", save_path, around_peak=True)
    make_combined_openfield_figure(save_path, 'combined_openfield_plot')


def plot_session_without_position(folder_path, df):
    print("I am now plotting signal without any position data...")
    file_utility.create_folder(folder_path, "/signal_plots")
    plot_signal(df, 'z_score', "Z-DF/F", folder_path + "/signal_plots")
    plot_signal(df, 'z_score', "Z-DF/F", folder_path + "/signal_plots", around_peak=True)
    make_combined_openfield_figure(folder_path + "/signal_plots", 'combined_signal_plot', position=False)


# format bout data for plotting heatmaps & signal per bout of object exploration
def create_object_df_for_plots(df):
    x_axis_length, y_axis_length = (7*30)+1, max(df.bout.values)+1
    times = np.arange(-2, 5, 7/x_axis_length)
    bouts = np.arange(1, y_axis_length, 1)
    axis_length = len(times)
    z_df = pd.DataFrame(columns=np.arange(-2, 5, 7/axis_length), index=np.arange(1, (max(bouts)+1)))  # NaNs
    count = 0

    for b in bouts:  # loop through bout data and populate new dataframe
        bout_df = df[df['bout'] == b]
        z_vals = bout_df.z_score.values
        z_df.iloc[count, 0:len(z_vals)] = z_vals
        count = count + 1

    return z_df, x_axis_length, y_axis_length


# plot heatmap of dF/F over time for each bout
def plot_bout_heatmap(df, save_path, save_name, ymin, ymax):
    try:
        df, x_axis_length, y_axis_length = create_object_df_for_plots(df)
        df = df.apply(pd.to_numeric)
        plt.figure(figsize=(7, 6), facecolor='w', edgecolor='k')
        ax = sns.heatmap(df, cmap='coolwarm', cbar=True, vmin=ymin, vmax=ymax)
        ax.set_xlabel("Time (s)", fontsize=22, labelpad=15)
        ax.set_ylabel("Bout", fontsize=22, labelpad=15)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        cbar.set_ticks([ymin, ymax])
        plt.xticks([0, 30, 60, 90, 120, 150, 180, x_axis_length],
                   ['-2', '-1', '0', '1', '2', '3', '4', '5'], fontsize=20, rotation=0)
        plt.yticks([y_axis_length - 1, 0], [str(int(y_axis_length) - 1), 1], fontsize=20)
        plt.axvline(60, 0, ymax, color='firebrick', linestyle='--', linewidth=3)
        plt.tight_layout()

    except ValueError:
        plt.figure(figsize=(7, 6), facecolor='w', edgecolor='k')

    plt.savefig(save_path + '/' + 'bout_heatmap_' + save_name + '.png', dpi=300)
    plt.savefig(save_path + '/' + 'bout_heatmap_' + save_name + '.eps', dpi=300)
    plt.close()


# plot signal against time for bouts of object exploration
def plot_signal_around_object(df, save_path, save_name, ymin, ymax):
    df['bout_str'] = df['bout'].astype(str)
    plt.figure(figsize=(7, 6))
    palette = sns.color_palette(['lightgrey'], len(df['bout_str'].unique()))
    ax = sns.lineplot(data=df, x="adjusted_time", y="z_score", hue='bout_str', palette=palette)
    ax2 = sns.lineplot(data=df, x="adjusted_time", y="z_score", color='black', linewidth=5, errorbar=None)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-2, 5)
    plt.axvline(0, 0, ymax, color='firebrick', linestyle='--', linewidth=3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_xlabel("Time (s)", fontsize=25, labelpad=15)
    ax.set_ylabel("Z-dF/F", fontsize=25, labelpad=15)
    ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig(save_path + '/' + 'smoothed_signal_' + save_name + '.png', dpi=300)
    plt.savefig(save_path + '/' + 'smoothed_signal_' + save_name + '.eps', dpi=300)
    plt.close()


def make_pivot(df, bins=20):
    if len(df) < 1:
        df.loc[len(df.columns)] = [0] * len(df.columns)  # add row of zeros so to avoid error
    df['x_bin'] = pd.cut(df.position_x, bins, include_lowest=True)  # bin positional data
    df['y_bin'] = pd.cut(df.position_y, bins, include_lowest=True)
    pivot = df.pivot_table(index='y_bin', columns='x_bin', values='z_score', aggfunc='mean')  # pivot table w bins
    pivot = pivot.iloc[::-1]  # make sure the bottom of the arena is the bottom of the arena - it defaults upside down
    return pivot


#  plot 2D heatmap of signal in the arena. Binned by 20x20 as a default (corresponds to 5x5cm in Level 6 openfield)
def plot_arena_heatmap(df, save_path, save_name, ymin, ymax, object_x=None, object_y=None, bins=20):
    plt.figure(figsize=(8, 6))
    pivot = make_pivot(df)
    ax = sns.heatmap(pivot, cmap='coolwarm', cbar=True, vmin=ymin, vmax=ymax)
    cbar = ax.collections[0].colorbar  # colorbar aesthetics
    cbar.ax.tick_params(labelsize=22)
    cbar.set_ticks([ymin, ymax])
    ax.set_xticks([0, bins/5, (bins/5)*2, (bins/5)*3, (bins/5)*4, bins])
    ax.set_yticks([0, bins/5, (bins/5)*2, (bins/5)*3, (bins/5)*4, bins])
    ax.set_xticklabels(['0', '20', '40', '60', '80', '100'], rotation=0, fontsize=22)
    ax.set_yticklabels(['100', '80', '60', '40', '20', '0'], fontsize=22)
    plt.xlabel("x (cm)", fontsize=26, labelpad=15)
    plt.ylabel("y (cm)", fontsize=26, labelpad=15)

    if object_x is not None:
        plt.scatter(object_x, object_y, s=1000, facecolors='none', edgecolors='firebrick')

    plt.title('Avg Z-dF/F in ' + str(int(100/bins)) + 'x' + str(int((100/bins))) + 'cm bins', fontsize=26)
    plt.tight_layout()
    plt.savefig(save_path + '/' + 'arena_heatmap_' + save_name + '.png', dpi=300)
    plt.savefig(save_path + '/' + 'arena_heatmap_' + save_name + '.eps', dpi=300)
    plt.close()


# combine object plots (openfield and bouts) into one figure
def make_combined_object_figure(path, filename, bouts=False):
    file_paths = file_utility.get_figure_paths(path, filename)  # get file paths
    if len(file_paths) > 0:
        plt.figure(figsize=(15, 18))
        grid = plt.GridSpec(2, 2)
        plot_to_grid(0, 0, grid, file_paths[0][0])  # trajectory
        plot_to_grid(0, 1, grid, file_paths[1][0])  # heatmap of arena

        if bouts:
            plot_to_grid(1, 0, grid, file_paths[2][0])  # heatmap of bouts
            plot_to_grid(1, 1, grid, file_paths[3][0])  # average signal

        plt.tight_layout()
        plt.savefig(path + '/combined_' + filename + '.png', dpi=300)
        plt.close()

    else:
        pass


# combine openfield plots for different session into one figure
def make_combined_session_figure(path):
    file_paths = file_utility.get_openfield_figure_paths(path)
    num_sessions = len(file_paths)
    if len(file_paths) > 1:
        plt.figure(figsize=(7.5*num_sessions, 12))
        grid = plt.GridSpec(2, num_sessions)
        for i in range(0, num_sessions):
            plot_to_grid(0, i, grid, file_paths[i][0][0])  # trajectory
            plot_to_grid(1, i, grid, file_paths[i][1][0])  # arena heatmap

        plt.tight_layout()
        plt.savefig(path + '/openfield_across_sessions.png', dpi=300)
        plt.close()


def make_object_plots(df, bout_df, path, filename, ylims, object_x=None, object_y=None):
    if len(df) > 0:
        file_utility.create_folder(path, '/object_plots/' + filename)  # folder for object plots
        folder_path = path + '/object_plots/' + filename
        plot_trajectory(df, folder_path, object_x, object_y, filename=filename)
        plot_arena_heatmap(df, folder_path, filename, ylims[0], ylims[1])

        if len(bout_df) > 0:  # only make these if there are exploration bouts
            plot_bout_heatmap(bout_df, folder_path, filename, ylims[2], ylims[3])
            plot_signal_around_object(bout_df, folder_path, filename, ylims[2], ylims[3])
            make_combined_object_figure(folder_path, filename, bouts=True)

        else:
            make_combined_object_figure(folder_path, filename)
