import sys
import os
from typing import Optional
from datetime import datetime
import pytz
import pandas as pd


from occident.utils import load_deepcell_object
from occident.velocity import (
    calculate_velocity_consecutive_frames,
    transform_velocity_df
)

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

sys.path.append(os.path.expanduser('.'))


def cell_tracking(
        test: bool,
        directory_path: str,
        save_data_root_path: str,
        group_logic: dict,
        window_size: Optional[int],
        time_between_frames: int,
        max_velocity_value: int,
):
    task_timestamp = datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d_%H:%M:%S')

    directory_path = os.path.expanduser(directory_path)

    filepaths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    codes = [code for sublist in group_logic.values() for code in sublist]
    filtered_filepaths = [filepath for filepath in filepaths if any(code in os.path.basename(filepath) for code in codes)]
    if test:
        print(f"***Loading test env***")
        filtered_filepaths = [filepath for filepath in filtered_filepaths if any(code in os.path.basename(filepath) for code in [
            'B3',
            'B7',
            'E3'
            ])]
        
    t_cell_velocity_dict = {}
    cancer_cell_velocity_dict = {}
    total_iterations = len(filtered_filepaths)
    for current_iteration, filepath in enumerate(filtered_filepaths, start=1):
        filename = os.path.basename(filepath)
        format_filename = filename.replace(".zip", "")
        print(f"\nIteration {current_iteration}/{total_iterations}")
        print(f"Processing data for file: {filename}")
        dcl_ob = load_deepcell_object(filepath)
        dcl_y = dcl_ob['y'][:,:,:,0,:]
        t_cell_array = dcl_y[0,:,:,:]
        cancer_cell_array = dcl_y[1,:,:,:]
        if test:
            print(f"***Loading test env***")
            t_cell_array = t_cell_array[0:15]
            cancer_cell_array = cancer_cell_array[0:15]
        print(f"T cell array shape: {t_cell_array.shape}")
        print(f"Cancer cell array shape: {cancer_cell_array.shape}")

        print(f"\nCalculating T cell velocities")
        if window_size is not None:
            t_cell_velocity_df = calculate_velocity_consecutive_frames(cell_array=t_cell_array, window_size=window_size, time_between_frames=time_between_frames, filename=filename)
        else: 
            t_cell_velocity_df = calculate_velocity_consecutive_frames(cell_array=t_cell_array, time_between_frames=time_between_frames, filename=filename)
        t_cell_velocity_dict[format_filename] = t_cell_velocity_df

        print(f"\nCalculating Cancer cell velocities")
        if window_size is not None:
            cancer_cell_velocity_df = calculate_velocity_consecutive_frames(cell_array=cancer_cell_array, window_size=window_size, time_between_frames=time_between_frames, filename=filename)
        else:
            cancer_cell_velocity_df = calculate_velocity_consecutive_frames(cell_array=cancer_cell_array, time_between_frames=time_between_frames, filename=filename)
        cancer_cell_velocity_dict[format_filename] = cancer_cell_velocity_df

        print("Processing dataframes")
        transformed_t_cell_velocity_dict = {}
        transformed_t_cell_velocity_dict = transform_velocity_df(t_cell_velocity_dict, transformed_t_cell_velocity_dict, max_velocity_value)
        transformed_cancer_cell_velocity_dict = {}
        transformed_cancer_cell_velocity_dict = transform_velocity_df(cancer_cell_velocity_dict, transformed_cancer_cell_velocity_dict, max_velocity_value)

    print(f"\nSaving Data")
    downloads_path = os.path.expanduser(f"{save_data_root_path}/cell_tracking_t_cell_csv_data_{task_timestamp}")
    if not os.path.exists(downloads_path):
        os.makedirs(downloads_path)
        print(f"Directory created at: {downloads_path}")
    else:
        print(f"Directory already exists at: {downloads_path}")
    for key, df in transformed_t_cell_velocity_dict.items():
        file_name = f"{key}.csv"
        full_path = os.path.join(downloads_path, file_name)
        df.to_csv(full_path, index=False)
        print(f"Saved {key} to {full_path}")
    
    downloads_path = os.path.expanduser(f"{save_data_root_path}/cell_tracking_cancer_csv_data_{task_timestamp}")
    if not os.path.exists(downloads_path):
        os.makedirs(downloads_path)
        print(f"Directory created at: {downloads_path}")
    else:
        print(f"Directory already exists at: {downloads_path}")
    for key, df in transformed_cancer_cell_velocity_dict.items():
        file_name = f"{key}.csv"
        full_path = os.path.join(downloads_path, file_name)
        df.to_csv(full_path, index=False)
        print(f"Saved {key} to {full_path}")

if __name__ == "__main__":
    test = False
    directory_path = '/gladstone/engelhardt/lab/MarsonLabIncucyteData/AnalysisFiles/CarnevaleRepStim/updated_full'
    save_data_root_path = './analysis'
    
    window_size = None
    time_between_frames = 1
    max_velocity_value = None
    group_logic = {
        "safe_harbor_ko": ["B3", "B4", "B5", "B6"],
        "cul5_ko": ["B7", "B8", "B9", "B10"],
        "rasa2_ko": ["E3", "E4", "E5", "E6"]
    }

   
    cell_tracking(
        test=test,
        time_between_frames=time_between_frames,
        max_velocity_value=max_velocity_value,
        window_size=window_size,
        directory_path=directory_path,
        save_data_root_path=save_data_root_path,
        group_logic=group_logic,
    )
