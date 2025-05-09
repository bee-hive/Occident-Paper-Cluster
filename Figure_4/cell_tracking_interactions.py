"""
- This script remove the first 50 frames from the analysis
- This script requires velocity data already generated from cell_tracking_velocity_data.py
"""

import sys
import os
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
from occident.utils import load_deepcell_object
from occident.tracking import (
    load_data_into_dataframe,
    process_all_frames,
    calculate_consecutive_frames,
    find_cell_interactions_with_counts,
    get_group_from_filename,
    filter_and_label
)

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

sys.path.append(os.path.expanduser('.'))

# helper functions that are specific to the tables and plots for the SH, RASA2, and CUL5 dataset
from cell_tracking_helper_functions import(
    make_roundness_plot_and_table,
    make_comparison_tables,
    make_linear_regression_tables,
    make_interaction_analysis_plots,
    plot_perimeter_area_roundness_velocity,
    plot_metrics_individual,
    run_linear_regression_tests
)

def cell_tracking(
        test: bool,
        min_consecutive_frames_list: list,
        post_interaction_windows: list,
        directory_path: str,
        existing_velocity_t_cell_data_path: str,
        existing_velocity_cancer_cell_data_path: str,
        group_logic: dict,
        colors: dict,
):
    task_timestamp = datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d_%H:%M:%S')
    directory_path = os.path.expanduser(directory_path)
    existing_velocity_t_cell_data_path = os.path.expanduser(existing_velocity_t_cell_data_path)
    existing_velocity_cancer_cell_data_path = os.path.expanduser(existing_velocity_cancer_cell_data_path)
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

    df_interactions_list = []
    df_consecutive_interactions_list = []
    t_cell_area_list = []
    cancer_cell_area_list = []
    total_iterations = len(filtered_filepaths)
    for current_iteration, filepath in enumerate(filtered_filepaths, start=1):
        filename = os.path.basename(filepath)
        print(f"\nIteration {current_iteration}/{total_iterations}")
        print(f"Processing data for file: {filename}")
        dcl_ob = load_deepcell_object(filepath)
        dcl_y = dcl_ob['y'][:,:,:,0,:]
        t_cell_array = dcl_y[0,:,:,:]
        cancer_cell_array = dcl_y[1,:,:,:]
        # removing first 50 frames:
        t_cell_array = t_cell_array[50:]
        cancer_cell_array = cancer_cell_array[50:]
        print(f"T cell array shape: {t_cell_array.shape}")
        print(f"Cancer cell array shape: {cancer_cell_array.shape}")

        print(f"\nSearching for Cell Interactions")
        df_interactions_file = find_cell_interactions_with_counts(t_cell_array, cancer_cell_array, filepath)
        df_interactions_file['interaction_id'] = df_interactions_file['t_cell_id'].astype(str) + '_' + df_interactions_file['cancer_cell_id'].astype(str)
        df_interactions_list.append(df_interactions_file)
        df_consecutive_interactions_file = calculate_consecutive_frames(df_interactions_file)
        df_consecutive_interactions_list.append(df_consecutive_interactions_file)
        print(f"Calculating T Cell Areas, Perimeters, and Roudness")
        t_cell_area_all_frames = process_all_frames(t_cell_array, filename)
        t_cell_area_all_frames.rename(columns={'cell_id': 't_cell_id', 'cell_area': 't_cell_area', 'cell_perimeter': 't_cell_perimeter'}, inplace=True)
        t_cell_area_all_frames['t_cell_roundness'] = 4 * np.pi * t_cell_area_all_frames['t_cell_area'] / (t_cell_area_all_frames['t_cell_perimeter'] ** 2)
        t_cell_area_list.append(t_cell_area_all_frames)
        print(f"Calculating Cancer Cell Areas, Perimeters, and Roudness")
        cancer_cell_area_all_frames = process_all_frames(cancer_cell_array, filename)
        cancer_cell_area_all_frames.rename(columns={'cell_id': 'cancer_cell_id', 'cell_area': 'cancer_cell_area', 'cell_perimeter': 'cancer_cell_perimeter'}, inplace=True)
        cancer_cell_area_all_frames['cancer_cell_roundness'] = 4 * np.pi * cancer_cell_area_all_frames['cancer_cell_area'] / (cancer_cell_area_all_frames['cancer_cell_perimeter'] ** 2)
        cancer_cell_area_list.append(cancer_cell_area_all_frames)
    
    print(f"\nCombining DataFrames")
    df_interactions = pd.concat(df_interactions_list, ignore_index=True)
    df_consecutive_interactions = pd.concat(df_consecutive_interactions_list, ignore_index=True)
    df_t_cell_area = pd.concat(t_cell_area_list, ignore_index=True)
    df_cancer_cell_area = pd.concat(cancer_cell_area_list, ignore_index=True)

    print(f"\nFiltering Out Problematic Cancer Cell Clumps")
    working_df = df_cancer_cell_area.copy()
    working_df['group'] = working_df['filename'].apply(lambda x: get_group_from_filename(x, group_logic))
    new_cancer_df = []
    for group in group_logic.keys():
        cancer_cell_group_data = working_df[working_df['group'] == group]
        cancer_cell_summary_stats = cancer_cell_group_data['cancer_cell_area'].describe()
        cancer_cell_upper_area_limit = cancer_cell_summary_stats['50%']*2.5
        new_group_data = cancer_cell_group_data[cancer_cell_group_data['cancer_cell_area'] <= cancer_cell_upper_area_limit]
        new_cancer_df.append(new_group_data)
    df_cancer_cell_area = pd.concat(new_cancer_df, ignore_index=True)
    
    print(f"\nLoading Existing Velocity Data")
    t_cell_downloads_path = os.path.expanduser(existing_velocity_t_cell_data_path)
    cancer_cell_downloads_path = os.path.expanduser(existing_velocity_cancer_cell_data_path)
    transformed_t_cell_velocity_df = load_data_into_dataframe(t_cell_downloads_path)
    transformed_cancer_cell_velocity_df = load_data_into_dataframe(cancer_cell_downloads_path)

    transformed_t_cell_velocity_df.rename(columns={'cell_id': 't_cell_id', 'velocity': 't_cell_velocity'}, inplace=True)
    transformed_cancer_cell_velocity_df.rename(columns={'cell_id': 'cancer_cell_id', 'velocity': 'cancer_cell_velocity'}, inplace=True)
    if test:
        relevant_filenames = df_interactions['filename'].unique().tolist()
        transformed_t_cell_velocity_df = transformed_t_cell_velocity_df[transformed_t_cell_velocity_df['filename'].isin(relevant_filenames)]
        transformed_cancer_cell_velocity_df = transformed_cancer_cell_velocity_df[transformed_cancer_cell_velocity_df['filename'].isin(relevant_filenames)]
    print(f"Converting units of velocity from pixels per frame to microns per frame")
    conversion_factor = 746 / 599
    dataframes = [transformed_t_cell_velocity_df, transformed_cancer_cell_velocity_df]
    for i, df in enumerate(dataframes):
        velocity_column = [col for col in df.columns if col.endswith('_velocity')]
        df[velocity_column] = df[velocity_column] * conversion_factor
        df = df[df['frame'] > 50]
        df['frame'] = df['frame'] - 50
        dataframes[i] = df
    transformed_t_cell_velocity_df, transformed_cancer_cell_velocity_df = dataframes

    df_consecutive_interactions_dict = {}
    for min_consecutive_frames in min_consecutive_frames_list:
        print(f"\nFiltering Consecutive Interactions with Minimum of {min_consecutive_frames} Frames")
        grouped_df_consecutive_interactions = df_consecutive_interactions.groupby(['unique_consec_group'])
        df_consecutive_interactions['interaction_id_max_consec_frames'] = grouped_df_consecutive_interactions['interaction_id_consec_frame'].transform('max')
        df_consecutive_interactions.reset_index(drop=True, inplace=True)
        filtered_df_consecutive_interactions = df_consecutive_interactions[df_consecutive_interactions['interaction_id_max_consec_frames'] >= min_consecutive_frames]
        filtered_df_consecutive_interactions['interaction_id_consec_frame'] = filtered_df_consecutive_interactions['interaction_id_consec_frame'] - 1
        print(f"Merging DataFrames")
        df_consecutive_interactions_merged = filtered_df_consecutive_interactions.merge(
            df_t_cell_area.drop(columns='perimeter_cell'), 
            on=['t_cell_id', 'frame', 'filename']
        ).merge(
            df_cancer_cell_area.drop(columns='perimeter_cell'), 
            on=['cancer_cell_id', 'frame', 'filename']
        )
        df_consecutive_interactions_merged = df_consecutive_interactions_merged.merge(
            transformed_t_cell_velocity_df, 
            on=['t_cell_id', 'frame', 'filename'], 
            how='left' 
        ).merge(
            transformed_cancer_cell_velocity_df, 
            on=['cancer_cell_id', 'frame', 'filename'], 
            how='left'
        )
        df_consecutive_interactions_merged['group'] = df_consecutive_interactions_merged['filename'].apply(lambda x: get_group_from_filename(x, group_logic))
        df_consecutive_interactions_dict[f'consec_{min_consecutive_frames}'] = df_consecutive_interactions_merged

    cell_id_type_list = ['t_cell', 'cancer_cell']
    selected_filename_list = df_interactions['filename'].unique().tolist()
    all_interactions_dict = {}
    for key in df_consecutive_interactions_dict.keys():
        filtered_frame_group = df_consecutive_interactions_dict[key]
        cell_type_dict = {}
        # Iterate over cell types
        for cell_id_type in cell_id_type_list:
            cell_type_columns = filtered_frame_group.filter(regex=f'^{cell_id_type}').columns.tolist()
            full_frame_list = []
            # Iterate over filenames
            for selected_filename in selected_filename_list:
                additional_columns1 = ['frame', 'filename', 'interaction_id_consec_frame', 'unique_consec_group']
                selected_filename_df = filtered_frame_group[filtered_frame_group['filename'] == selected_filename][cell_type_columns + additional_columns1]
                selected_unique_consec_group_list = selected_filename_df['unique_consec_group'].unique().tolist()
                for selected_unique_consec_group in selected_unique_consec_group_list:
                    unique_consec_group_selected_filename_df = selected_filename_df[selected_filename_df['unique_consec_group'] == selected_unique_consec_group]
                    if 'cancer_cell' in cell_id_type:
                        selected_full_cell_info_df = df_cancer_cell_area[df_cancer_cell_area['filename'] == selected_filename]
                        selected_full_cell_info_df = selected_full_cell_info_df.merge(
                            transformed_cancer_cell_velocity_df, 
                            on=['cancer_cell_id', 'frame', 'filename'], 
                            how='left'
                        )
                    if 't_cell' in cell_id_type:
                        selected_full_cell_info_df = df_t_cell_area[df_t_cell_area['filename'] == selected_filename]
                        selected_full_cell_info_df = selected_full_cell_info_df.merge(
                            transformed_t_cell_velocity_df, 
                            on=['t_cell_id', 'frame', 'filename'], 
                            how='left' 
                        )
                    # filtered_frame_group.filter(regex=f'^{cell_id_type}').columns.tolist()
                    cell_type_id_columns = unique_consec_group_selected_filename_df.filter(regex=f'(.*{cell_id_type}.*id)|(.*id.*{cell_id_type})').columns.tolist()
                    additional_columns2 = ['frame', 'interaction_id_consec_frame', 'unique_consec_group']
                    cell_id_first_frame_pair = unique_consec_group_selected_filename_df[unique_consec_group_selected_filename_df['interaction_id_consec_frame'] == 0][cell_type_id_columns + additional_columns2]
        
                    merged_df = pd.merge(cell_id_first_frame_pair, selected_full_cell_info_df, on=cell_type_id_columns[0], suffixes=('_x', '_y'))
                    negative_frame_set = merged_df.groupby(cell_type_id_columns[0]).apply(filter_and_label, include_groups=True).reset_index(drop=True)
                    negative_frame_set.drop(columns=['frame_x'], inplace=True)
                    negative_frame_set.rename(columns={'frame_y': 'frame'}, inplace=True)
                    negative_frame_set = negative_frame_set[cell_type_columns + additional_columns1]
                    clipped_unique_consec_group_selected_filename_df = unique_consec_group_selected_filename_df[unique_consec_group_selected_filename_df['interaction_id_consec_frame'].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
                    full_frame_set = pd.concat([clipped_unique_consec_group_selected_filename_df, negative_frame_set], ignore_index=True)
                    full_frame_set.drop_duplicates(inplace=True)
                    full_frame_set['group'] = full_frame_set['filename'].apply(lambda x: get_group_from_filename(x, group_logic))
                    full_frame_set.sort_values(by=[cell_type_id_columns[0], 'unique_consec_group', 'frame'], inplace=True)
                    # full_frame_set[full_frame_set['cell_type_id_columns'] == 147]
                    full_frame_list.append(full_frame_set)
            # Store the aggregated data for the current cell type
            cell_type_dict[cell_id_type] = pd.concat(full_frame_list, ignore_index=True)
        # Store results for the current key
        all_interactions_dict[key] = cell_type_dict

    # Process for 't_cell'
    t_cell_frames = []
    for consec_key in all_interactions_dict.keys():
        # Check if the 't_cell' key exists in the sub-dictionary
        if 't_cell' in all_interactions_dict[consec_key]:
            df = all_interactions_dict[consec_key]['t_cell'].copy()
            df['consec_frame_group'] = consec_key  # Add new column with the group key
            t_cell_frames.append(df)
    # Concatenate all DataFrames for 't_cell'
    t_cell_combined = pd.concat(t_cell_frames, ignore_index=True)
    # Process for 'cancer_cell'
    cancer_cell_frames = []
    for consec_key in all_interactions_dict.keys():
        # Check if the 'cancer_cell' key exists in the sub-dictionary
        if 'cancer_cell' in all_interactions_dict[consec_key]:
            df = all_interactions_dict[consec_key]['cancer_cell'].copy()
            df['consec_frame_group'] = consec_key  # Add new column with the group key
            cancer_cell_frames.append(df)
    # Concatenate all DataFrames for 'cancer_cell'
    cancer_cell_combined = pd.concat(cancer_cell_frames, ignore_index=True)
    cell_combined_dict = {
        't_cell': t_cell_combined,
        'cancer_cell': cancer_cell_combined
    }

    print(f"\nMaking T-Cell Roundness Plot and Table")
    base_save_roundness_plot_filename = './plots/average_t_cell_roundness_{task_timestamp}_past50.pdf'
    base_save_roundness_table_filename = './tables/t_cell_group_roundness_summary_{task_timestamp}_past50.csv'
    base_save_roundness_plot_filename = os.path.expanduser(base_save_roundness_plot_filename)
    base_save_roundness_table_filename = os.path.expanduser(base_save_roundness_table_filename)
    
    make_roundness_plot_and_table(
        input_df=df_t_cell_area,
        group_logic=group_logic,
        colors=colors,
        task_timestamp=task_timestamp,
        base_save_roundness_table_filename=base_save_roundness_table_filename,
        base_save_roundness_plot_filename=base_save_roundness_plot_filename,
    )
    
    print(f"\nRunning Tests")
    base_t_cell_comparison_tbl_filename = "./tables/t_cell_{consec_frame_group}_results_{base_group}_{task_timestamp}_past50.csv"
    base_cancer_cell_comparison_tbl_filename = "./tables/cancer_cell_{consec_frame_group}_results_{base_group}_{task_timestamp}_past50.csv"
    base_t_cell_linear_regression_filename = './tables/t_cell_linear_regression_{task_timestamp}_past50.csv'
    base_cancer_cell_linear_regression_filename = './tables/cancer_cell_linear_regression_{task_timestamp}_past50.csv'
    base_interaction_distribution_plot_filename = "./plots/interaction_distribution_plot_{task_timestamp}_past50.pdf"
    base_interaction_distribution_table_filename = "./tables/interaction_distribution_table_{task_timestamp}_past50.csv"
    base_interaction_distribution_over_time_plot_filename = "./plots/interaction_distribution_plot_over_time_{task_timestamp}_past50.pdf"
    base_unique_interaction_table_filename = "./tables/unique_interaction_table_{task_timestamp}_past50.csv"
    base_linear_regression_model_summary_table = './tables/linear_regression_model_summary_{task_timestamp}.csv'

    base_t_cell_comparison_tbl_filename = os.path.expanduser(base_t_cell_comparison_tbl_filename)
    base_cancer_cell_comparison_tbl_filename = os.path.expanduser(base_cancer_cell_comparison_tbl_filename)
    base_t_cell_linear_regression_filename = os.path.expanduser(base_t_cell_linear_regression_filename)
    base_cancer_cell_linear_regression_filename = os.path.expanduser(base_cancer_cell_linear_regression_filename)
    base_interaction_distribution_plot_filename = os.path.expanduser(base_interaction_distribution_plot_filename)
    base_interaction_distribution_table_filename = os.path.expanduser(base_interaction_distribution_table_filename)
    base_interaction_distribution_over_time_plot_filename = os.path.expanduser(base_interaction_distribution_over_time_plot_filename)
    base_unique_interaction_table_filename = os.path.expanduser(base_unique_interaction_table_filename)
    base_linear_regression_model_summary_table = os.path.expanduser(base_linear_regression_model_summary_table)
    
    make_comparison_tables(
        input_df=cell_combined_dict['t_cell'], 
        task_timestamp=task_timestamp, 
        base_filename=base_t_cell_comparison_tbl_filename
        )
    make_comparison_tables(
        input_df=cell_combined_dict['cancer_cell'], 
        task_timestamp=task_timestamp, 
        base_filename=base_cancer_cell_comparison_tbl_filename
        )
    make_linear_regression_tables(
        input_df=cell_combined_dict['t_cell'],
        task_timestamp=task_timestamp,
        base_linear_regression_filename=base_t_cell_linear_regression_filename
    )
    make_linear_regression_tables(
        input_df=cell_combined_dict['cancer_cell'],
        task_timestamp=task_timestamp,
        base_linear_regression_filename=base_cancer_cell_linear_regression_filename
    )
    run_linear_regression_tests(
        cell_combined_dict=cell_combined_dict, 
        task_timestamp=task_timestamp, 
        base_linear_regression_model_summary_table=base_linear_regression_model_summary_table
    )
    make_interaction_analysis_plots(
        df_consecutive_interactions_dict=df_consecutive_interactions_dict,
        colors=colors,
        task_timestamp=task_timestamp,
        base_unique_interaction_table_filename=base_unique_interaction_table_filename,
        base_interaction_distribution_table_filename=base_interaction_distribution_table_filename,
        base_interaction_distribution_plot_filename=base_interaction_distribution_plot_filename,
        base_interaction_distribution_over_time_plot_filename=base_interaction_distribution_over_time_plot_filename
    )

    t_cell_plot_filename = f'./plots/t_cell_perimeter_area_roundness_velocity_plots_{task_timestamp}_past50.pdf'
    cancer_cell_plot_filename = f'./plots/cancer_cell_perimeter_area_roundness_velocity_plots_{task_timestamp}_past50.pdf'  
    t_cell_plot_base_filename_individual = './plots/'
    cancer_cell_plot_base_filename_individual = './plots/'

    t_cell_plot_filename = os.path.expanduser(t_cell_plot_filename)
    cancer_cell_plot_filename = os.path.expanduser(cancer_cell_plot_filename)
    t_cell_plot_base_filename_individual = os.path.expanduser(t_cell_plot_base_filename_individual)
    cancer_cell_plot_base_filename_individual = os.path.expanduser(cancer_cell_plot_base_filename_individual)

    plot_perimeter_area_roundness_velocity(cell_combined_dict['t_cell'], colors, t_cell_plot_filename)
    plot_perimeter_area_roundness_velocity(cell_combined_dict['cancer_cell'], colors, cancer_cell_plot_filename)
    plot_metrics_individual(cell_combined_dict['t_cell'], colors, t_cell_plot_base_filename_individual, task_timestamp)
    plot_metrics_individual(cell_combined_dict['cancer_cell'], colors, cancer_cell_plot_base_filename_individual, task_timestamp)

if __name__ == "__main__":
    test = False
    directory_path = '/gladstone/engelhardt/lab/MarsonLabIncucyteData/AnalysisFiles/CarnevaleRepStim/updated_full/'
    # use paths that are generated by cell_tracking_velocity_data.py
    # make sure the time stamp at the end matches the correct run of cell_tracking_velocity_data.py
    existing_velocity_t_cell_data_path = '/gladstone/engelhardt/lab/adamw/Occident-Paper-Cluster/analysis/cell_tracking_t_cell_csv_data_2025-03-03_11:44:03'
    existing_velocity_cancer_cell_data_path = '/gladstone/engelhardt/lab/adamw/Occident-Paper-Cluster/analysis/cell_tracking_cancer_csv_data_2025-03-03_11:44:03'

    group_logic = {
        "safe_harbor_ko": ["B3", "B4", "B5", "B6"],
        "cul5_ko": ["B7", "B8", "B9", "B10"],
        "rasa2_ko": ["E3", "E4", "E5", "E6"]
    }
    colors = {
        'Safe Harbor KO': '#a9a9a9', 
        'RASA2 KO': '#800000',
        'CUL5 KO': '#000075'
    }
    
    min_consecutive_frames_list = [2, 5, 10]
    post_interaction_windows = [5, 10]

    cell_tracking(
        test=test,
        min_consecutive_frames_list=min_consecutive_frames_list,
        post_interaction_windows=post_interaction_windows,
        directory_path=directory_path,
        existing_velocity_t_cell_data_path=existing_velocity_t_cell_data_path,
        existing_velocity_cancer_cell_data_path=existing_velocity_cancer_cell_data_path,
        group_logic=group_logic,
        colors=colors,
    )