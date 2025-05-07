"""
Script for processing fiber photometry data collected using the pyPhotometry system

Expects data in the following format:
    Mouse_OF_YYYY-MM-DD for photometry (.ppd) and position data (.csv), ie 1396_OF_2022-03-28
    For objects: positions in a .csv file containing string 'object', ie 1396_OF_2022-03-28_object.csv
                 Expects separate .csv file for each object detected

Steps:
    1. Remove noise, photobleaching and movement artifacts
    2. Retrieve speed, heading direction, and position of the mouse in an open-field
    3. Synchronise mouse position with signal using an LED sync pulse
    4. Generate plots of signal vs time for the whole session and 10s around peak signal
    5. If there are object files, analyses signal for each bout of object exploration

Outputs:
    Plots of preprocessing steps (Step 1) saved to /preprocessing_plots
    Plots of signal versus time with animal trajectory are saved to /openfield_plots (Step 4)
    If no position data, plots of signal versus time are saved to /signal_plots (Step 4)
    If object files found, plots of signal around object are saved to /object_plots (Step 5)
    Dataframes containing raw and processed data are saved as .pkl files
"""
import file_utility
import synchronise_data
import preprocessing_photometry_data
import process_position_data
import plotting_utility
import process_object_session
import os


def check_session_type(metadata_path, recording):
    with open(metadata_path, 'r') as file:
        for line in file:
            if line.strip() == f'session_type=odor_vector' or line.strip() == f'session_type=object_vector':
                process_photometry_data(recording)
            else:
                pass


def control_analysis(file, single_run=False):
    """
    Run analysis pipeline on single or multiple recordings
    :param file: path to .txt file containing list of recordings as paths
    :param single_run: if True, run analysis on a single recording specified to file argument
    """
    if single_run:
        process_photometry_data(file)

    else:
        try:
            recordings = file_utility.read_recording_list(file)
            print("I have found a list of", len(recordings), "recordings. I will analyse these now...")

            for recording in recordings:
                print("  ")
                print('----------------------------------------------------------------------------------------')
                print("I am now processing:", recording)

                try:
                    path = os.path.join(recording, 'metadata.txt')
                    check_session_type(path, recording)
                   # process_photometry_data(recording)

                    print(recording, 'was processed successfully!')

                except:
                    print("There was a problem with this recording. I will move onto the next one.")
                    path = file.rsplit('/', 1)[0]
                    with open(path + "/crashlist.txt", 'a') as f:
                        f.write(recording + '\n')

        except FileNotFoundError:
            print("I could not find this file:", file, ". Analysis will not proceed.")


def process_photometry_data(path):
    """
    :param path: path to recording folder. Outputs saved in this folder.
    :return: dataframe containing processed signal, animal position (if found), and timestamps
    """
    file_paths = file_utility.get_file_paths(path)  # find data files
    raw_df, pp_df = preprocessing_photometry_data.preprocessing(path, file_paths[1][0])  # photometry
    skip_object_analysis = False

    if not os.path.exists(path + "/dataframes"):
        os.mkdir(path + "/dataframes")

    try:
        is_found, position, zero_ref, xmin, ymin = process_position_data.analyse_position(file_paths[0][0])  # position

    except IndexError:  # handle index error if there is no position file
        print('There was no position file detected for this recording.')
        is_found = False

    # if position found, sync position & photometry data
    if is_found:
        try:
            position, pp_df, lag = synchronise_data.synchronise_data(pp_df, position, path)  # synchronise
            processed_df = synchronise_data.merge_synchronised_data(pp_df, position)  # merge into one dataframe
            plotting_utility.plot_open_field_session(path, processed_df)  # make openfield plots

        # handle where there is an error with synchronisation due to mismatched lengths
        except (AssertionError, IndexError) as error:
            print('There was a problem during syncing. These data cannot be synchronised.')
            print('This is what Python said happened: ')
            print(error)
            # process data without position
            processed_df = pp_df
            plotting_utility.plot_session_without_position(path, processed_df)
            skip_object_analysis = True  # do not try to process object positions

        if skip_object_analysis is False:
            # check if there were objects in the session
            object_found, object_files, num_objects = file_utility.check_for_object_files(path)

            # if there were objects, analyse signal around objects
            if object_found:
                print('----------------------------------------------------------------------------------------')
                print('This is an object session. I will analyse object exploration now.')
                print('----------------------------------------------------------------------------------------')
                print('I am processing object positions...')
                print(f'The positions of {num_objects} object(s) were found for this recording.')

                # default settings: bouts are detected when the animal is within 3 cm of object edge for > 1s
                process_object_session.analyse_object_session(processed_df, path, object_files, zero_ref, lag, xmin, ymin)

    # process signal without position if there was no position file
    else:
        processed_df = pp_df
        plotting_utility.plot_session_without_position(path, processed_df)

    # save dataframes
    raw_df.to_pickle(path + "//dataframes/raw_data.pkl")
    processed_df.to_pickle(path + "/dataframes/processed_data.pkl")


def main():
    # to run analysis on list of recordings, put path to .txt file containing path list here:
     control_analysis("/Users/brivandrey/Desktop/FP/1618/recordings.txt")

    # to run analysis on a single recording, put path to the recording folder here and set single_run to True
    #control_analysis("/Users/brivandrey/Desktop/FP/1541/1541_2023-03-07-143611", single_run=True)


if __name__ == '__main__':

    main()
