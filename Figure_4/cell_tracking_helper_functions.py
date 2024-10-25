import os
import sys
import re
import json
from typing import Optional
import socket
from datetime import datetime
import pytz
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from io import BytesIO
import scipy.stats
import itertools
import math
import tarfile
from scipy.ndimage import find_objects
from scipy.ndimage import label
from scipy.stats import sem, ttest_ind_from_stats, norm
from scipy.stats import linregress
from skimage.morphology import square, binary_erosion, binary_dilation
from skimage.morphology import remove_small_objects
import statsmodels.api as sm
import statsmodels.formula.api as smf
import skimage as sk
import zipfile
import tifffile

def load_data_local(
        filepath
):
    f = zipfile.ZipFile(filepath, 'r')
    file_bytes = f.read("cells.json")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        cells = json.load(b)
    file_bytes = f.read("divisions.json")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        divisions = json.load(b)
    file_bytes = f.read("X.ome.tiff")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        X = sk.io.imread(b, plugin="tifffile")
    file_bytes = f.read("y.ome.tiff")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        y = sk.io.imread(b, plugin="tifffile")
    dcl_ob = {
        'X': np.expand_dims(X,3),
        'y': np.expand_dims(y,3),
        'divisions':divisions,
        'cells': cells}
    return dcl_ob

def load_data_into_dataframe(data_path):
    # Initialize an empty DataFrame to store concatenated results
    combined_df = pd.DataFrame()
    # Expand the path to the directory
    downloads_path = os.path.expanduser(data_path)
    # Iterate over each file in the directory
    for file_name in os.listdir(downloads_path):
        if file_name.endswith('.csv'):
            # Construct the full path to the file
            full_path = os.path.join(downloads_path, file_name)
            # Load the CSV file into a DataFrame
            df = pd.read_csv(full_path)
            # Concatenate the loaded DataFrame with the combined DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Remove '.zip' from all rows in the 'filename' column if the column exists
    if 'filename' in combined_df.columns:
        combined_df['filename'] = combined_df['filename'].str.replace('.zip', '', regex=False)
    
    combined_df['frame'] = combined_df['frame'] + 1

    return combined_df

def analyze_cells(frame):
    unique_cells = np.unique(frame)
    unique_cells = unique_cells[unique_cells != 0]
    
    conversion_factor = 746 / 599
    area_conversion_factor = conversion_factor ** 2  # Converting area to microns squared
    
    cell_data = {'cell_id': [], 'cell_area': [], 'perimeter_cell': [], 'cell_perimeter': []}
    for cell_id in unique_cells:
        cell_area_pixels = np.sum(frame == cell_id)
        cell_area = cell_area_pixels * area_conversion_factor  # Convert area from pixels^2 to microns^2
        
        on_perimeter = np.any(frame[0, :] == cell_id) or np.any(frame[:, 0] == cell_id) or \
                       np.any(frame[-1, :] == cell_id) or np.any(frame[:, -1] == cell_id)
        
        cell_perimeter_pixels = calculate_perimeter(frame, cell_id)
        cell_perimeter = cell_perimeter_pixels * conversion_factor  # Convert perimeter from pixels to microns
        
        cell_data['cell_id'].append(cell_id)
        cell_data['cell_area'].append(cell_area)
        cell_data['perimeter_cell'].append(on_perimeter)
        cell_data['cell_perimeter'].append(cell_perimeter)
    
    cell_data_df = pd.DataFrame(cell_data)
    return cell_data_df

def calculate_perimeter(frame, cell_id):
    # Create a binary mask where the current cell_id is 1, and others are 0
    cell_mask = frame == cell_id

    # Pad the mask with zeros on all sides to handle edge cells correctly
    padded_mask = np.pad(cell_mask, pad_width=1, mode='constant', constant_values=0)

    # Count transitions from 1 to 0 (cell to non-cell) at each pixel
    perimeter = (
        np.sum(padded_mask[:-2, 1:-1] & ~padded_mask[1:-1, 1:-1]) +  # up
        np.sum(padded_mask[2:, 1:-1] & ~padded_mask[1:-1, 1:-1]) +   # down
        np.sum(padded_mask[1:-1, :-2] & ~padded_mask[1:-1, 1:-1]) +  # left
        np.sum(padded_mask[1:-1, 2:] & ~padded_mask[1:-1, 1:-1])     # right
    )

    return perimeter

def process_all_frames(
        input_array, 
        filename,
        use_connected_component_labeling: Optional[bool] = False
):
    all_frames_data = []
    
    # Extract the start frame number from the filename using a regular expression
    match = re.search(r'start_(\d+)_end_(\d+)', filename)
    if match:
        start_frame = int(match.group(1))

    # Iterate through each frame in all_tcell_mask
    for frame_idx in range(input_array.shape[0]):
        frame = input_array[frame_idx, :, :]

        if use_connected_component_labeling:
            # Label the connected components in the frame
            frame, num_features = label(frame)
            print(f"Frame {frame_idx + 1} has {num_features} connected components")

        cell_data_df = analyze_cells(frame)  # Ensure analyze_cells accepts filename
        # Calculate the correct frame number based on the start frame
        cell_data_df['frame'] = start_frame + frame_idx + 1
        cell_data_df['filename'] = filename.replace('.zip', '')
        all_frames_data.append(cell_data_df)

    # Concatenate all DataFrames in the list into one large pd DataFrame
    all_cell_data_df = pd.concat(all_frames_data, ignore_index=True)

    return all_cell_data_df

def calculate_centroid(
        matrix, 
        cell_id
):
    # calculate the centroid
    y, x = np.where(matrix == cell_id)
    if len(x) == 0 or len(y) == 0:
        return None
    return np.mean(x), np.mean(y)

def calculate_centroids_for_all_cells(frame, unique_cell_ids):
    centroids = {}
    for cell_id in unique_cell_ids:
        y, x = np.where(frame == cell_id)
        if len(x) > 0 and len(y) > 0:
            centroids[cell_id] = (np.mean(x), np.mean(y))
        else:
            centroids[cell_id] = None
    return centroids

def calculate_velocity_consecutive_frames(
        cell_array,
        time_between_frames,
        filename
):
    n_frames = cell_array.shape[0]
    unique_cell_ids = np.unique(cell_array[cell_array != 0])  # Exclude background ID

    df_velocity = pd.DataFrame(unique_cell_ids, columns=['cell_id'])
    velocities = {}

    for frame_index in range(1, n_frames):  # Start from frame index 1 (not 0) to compare with t-1
        if frame_index == 1 or frame_index % 10 == 0:
            print(f"Processing Frame: {frame_index} out of {n_frames}")
        column_name = f'v_frame_{frame_index-1}_to_{frame_index}'
        velocities[column_name] = np.nan * np.ones(len(unique_cell_ids))  # Initialize column with NaNs

        centroids_t = calculate_centroids_for_all_cells(cell_array[frame_index, :, :], unique_cell_ids)
        centroids_t_minus_1 = calculate_centroids_for_all_cells(cell_array[frame_index - 1, :, :], unique_cell_ids)

        for cell_id in unique_cell_ids:
            centroid_t = centroids_t.get(cell_id)
            centroid_t_minus_1 = centroids_t_minus_1.get(cell_id)

            if centroid_t is None or centroid_t_minus_1 is None:
                continue  # Skip if centroids can't be calculated

            displacement = np.linalg.norm(np.array(centroid_t) - np.array(centroid_t_minus_1))
            velocity = displacement / time_between_frames
            
            cell_index = df_velocity[df_velocity['cell_id'] == cell_id].index.item()
            velocities[column_name][cell_index] = velocity

    # Add velocities to the DataFrame
    for key, value in velocities.items():
        df_velocity[key] = value
    df_velocity['filename'] = filename.replace('.zip', '')

    return df_velocity

def transform_velocity_df(
        velocity_dict, 
        transformed_dict,
        max_velocity_value
):
    for key, df in velocity_dict.items():
        # Melt the DataFrame
        melted_df = pd.melt(df, id_vars=['cell_id', 'filename'], var_name='frame', value_name='velocity',
                            value_vars=[col for col in df.columns if col.startswith('v_frame_')])
        # Remove rows where 'velocity' is NaN and >= max_velocity_value
        filtered_df = melted_df.dropna(subset=['velocity']).copy()
        if max_velocity_value is not None:
            filtered_df = filtered_df[filtered_df['velocity'] < max_velocity_value]
        # Reset index
        filtered_df.reset_index(drop=True, inplace=True)
        # Adjust the 'frame' column to be an integer by extracting the first number in the naming pattern
        filtered_df['frame'] = filtered_df['frame'].str.extract(r'(\d+)_to_').astype(int)
        # Sort by 'cell_id' and 'frame'
        filtered_df.sort_values(by=['frame', 'cell_id'], inplace=True)
        # Store the transformed DataFrame in the new dictionary
        transformed_dict[key] = filtered_df
    return transformed_dict

def combine_dataframes(
        dataframe_dict, 
        group_logic
):
    combined_dataframe_dict = {}
    # Iterate over each group and its codes in group_logic
    for group_name, codes in group_logic.items():
        dfs_for_group = []
        
        # Iterate over each code in the current group
        for code in codes:
            # Filter keys (filenames) in dataframe_dict by current code
            for filename in dataframe_dict:
                if code in filename:
                    dfs_for_group.append(dataframe_dict[filename])
        
        # Combine all DataFrames in the list into a single DataFrame
        if dfs_for_group:
            combined_df = pd.concat(dfs_for_group, ignore_index=True)
            combined_df.sort_values('frame', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
            combined_dataframe_dict[group_name] = combined_df
    
    return combined_dataframe_dict

def make_segmentation_plot(
        title_prefix,
        remove_perimeter_cell,
        combined_dataframe_dict, 
        plot_group_colors, 
        test,
        plot_name,
        task_timestamp,
        save_plot_path
):
    plt.figure(figsize=(14, 7))
    color_iter = iter(plot_group_colors)  # Create an iterator over the colors list

    for group_name, df_for_plot in combined_dataframe_dict.items():
        if remove_perimeter_cell:
            df_for_plot = df_for_plot[df_for_plot['perimeter_cell'] == False]

        # Calculate mean cell_area for every individual frame
        mean_cell_area_per_frame = df_for_plot.groupby('frame')['cell_area'].mean()
        
        # Calculate SEM for cell_area for every individual frame
        sem_cell_area_per_frame = df_for_plot.groupby('frame')['cell_area'].apply(sem)

        # Get the next color from the iterator
        color = next(color_iter, 'gray')  # Default to 'gray' if the colors list is exhausted

        # Plot the mean cell_area per frame with a line
        plt.plot(mean_cell_area_per_frame.index, mean_cell_area_per_frame, lw=2, color=color, label=f'{group_name} Mean Cell Area')

        # Add the error area for SEM
        plt.fill_between(mean_cell_area_per_frame.index, mean_cell_area_per_frame - sem_cell_area_per_frame, 
                         mean_cell_area_per_frame + sem_cell_area_per_frame, color=color, alpha=0.2, label=f'{group_name} Error')

    if remove_perimeter_cell:
        title_prefix = f"{title_prefix} (Excluding Perimeter Cells)"
    plt.title(f'{title_prefix}: Mean Cell Area per Frame')
    plt.xlabel('Frame')
    plt.ylabel('Mean Cell Area')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()

    if remove_perimeter_cell:
        plot_name = f"{plot_name}_filtered_perimeter"

    
    plot_path_name = os.path.join(save_plot_path, f"{plot_name}_{task_timestamp}.pdf")

    plt.savefig(plot_path_name)
    plt.close()  # Close the figure to free memory
    print(f"Plot saved: {plot_path_name}")

def safe_sem(x):
    return sem(x) if len(x) > 1 else np.NaN  # Return NaN if not enough data

def make_velocity_plot(
        group_type,
        title_prefix,
        combined_mask_dict, 
        plot_group_colors, 
        test,
        plot_name,
        task_timestamp,
        save_plot_path
):
    plt.figure(figsize=(14, 7))
    color_iter = iter(plot_group_colors)  # Create an iterator over the colors list

    if group_type == 'individual':
        print(f"Grouping by individual frame")
        for group_name, df_for_plot in combined_mask_dict.items():

            print(f"Processing: {group_name}")
            # Calculate mean velocity for every individual frame
            mean_velocity_per_frame = df_for_plot.groupby('frame')['velocity'].mean()
            
            # Calculate SEM for velocity for every individual frame
            sem_velocity_per_frame = df_for_plot.groupby('frame')['velocity'].apply(safe_sem)

            # Get the next color from the iterator
            color = next(color_iter, 'gray')  # Default to 'gray' if the colors list is exhausted

            # Plot the mean velocity per frame with a line
            plt.plot(mean_velocity_per_frame.index, mean_velocity_per_frame, lw=2, color=color, label=f'{group_name} Mean Velocity')

            # Add the error area for SEM
            plt.fill_between(mean_velocity_per_frame.index, mean_velocity_per_frame - sem_velocity_per_frame, 
                            mean_velocity_per_frame + sem_velocity_per_frame, color=color, alpha=0.2, label=f'{group_name} Error')

        plt.title(f'{title_prefix}: Mean Velocity per Frame')
        plt.xlabel('Frame')
        plt.ylabel('Mean Velocity')
        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()

        plot_path_name = os.path.join(save_plot_path, f"{plot_name}_{group_type}_{task_timestamp}.pdf")

        plt.savefig(plot_path_name)
        plt.close()  # Close the figure to free memory
        print(f"Plot saved: {plot_path_name}")
    if group_type == 'combined':
        print(f"Grouping by frame group")
        for group_name, df_for_plot in combined_mask_dict.items():
            print(f"Processing: {group_name}")
            # Create a 'frame group' column for grouping frames into sets of 50
            df_for_plot['frame_group'] = df_for_plot['frame'] // 50

            # Calculate mean velocity for every 'frame group'
            mean_velocity_per_frame_group = df_for_plot.groupby('frame_group')['velocity'].mean()
            
            # Calculate SEM for velocity for every 'frame group'
            sem_velocity_per_frame_group = df_for_plot.groupby('frame_group')['velocity'].apply(lambda x: sem(x) if len(x) > 1 else np.NaN)

            # Get the next color from the iterator
            color = next(color_iter, 'gray')  # Default to 'gray' if the colors list is exhausted

            # Plot the mean velocity per 'frame group' with a line
            plt.plot(mean_velocity_per_frame_group.index * 50, mean_velocity_per_frame_group, lw=2, color=color, label=f'{group_name} Mean Velocity')

            # Add the error area for SEM
            plt.fill_between(mean_velocity_per_frame_group.index * 50, mean_velocity_per_frame_group - sem_velocity_per_frame_group, 
                            mean_velocity_per_frame_group + sem_velocity_per_frame_group, color=color, alpha=0.2, label=f'{group_name} Error')

        plt.title(f'{title_prefix}: Mean Velocity per Frame Group')
        plt.xlabel('Frame Group Start')
        plt.ylabel('Mean Velocity')
        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()

        plot_path_name = os.path.join(save_plot_path, f"{plot_name}_{group_type}_{task_timestamp}.pdf")

        plt.savefig(plot_path_name)
        plt.close()  # Close the figure to free memory
        print(f"Plot saved: {plot_path_name}")

def make_violin_plot(
        test,
        combined_mask_dict,
        mask_name,
        plot_name,
        task_timestamp,
        save_plot_path_violin
):
    # Create an empty DataFrame
    all_data = pd.DataFrame()
    
    # Append all groups data into one DataFrame and add a 'Group' column
    for group_name, df in combined_mask_dict.items():
        df['Group'] = group_name
        all_data = pd.concat([all_data, df], ignore_index=True)

    # Setup the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Creating violin plot on the specified Axes
    sns.violinplot(ax=ax, x='Group', y='velocity', data=all_data)
    
    mask_name_title = mask_name.title().replace('_', ' ')
    ax.set_title(f'Velocity Distribution Across Groups - {mask_name_title}')
    ax.set_xlabel('Group')
    ax.set_ylabel('Velocity')
    
    plt.tight_layout()
    
    plot_path_name = os.path.join(save_plot_path_violin, f"{plot_name}_{task_timestamp}.pdf")
    
    plt.savefig(plot_path_name)
    plt.close(fig)  # Close the figure to free memory
    print(f"Plot saved: {plot_path_name}")

def identify_consecutive_frames(interaction_id_grouped):
    # Calculate the difference between current and previous frames
    interaction_id_grouped['frame_diff'] = interaction_id_grouped['frame'] - interaction_id_grouped['frame'].shift(1, fill_value=interaction_id_grouped['frame'].iloc[0])
    
    # Identify where frames are consecutive
    interaction_id_grouped['is_consecutive'] = interaction_id_grouped['frame_diff'] == 1
    
    # Create unique identifiers for consecutive frame sequences
    interaction_id_grouped['consec_group'] = (~interaction_id_grouped['is_consecutive']).cumsum()

    # Extract the part of the filename between the first and second underscore
    interaction_id_grouped['file_segment'] = interaction_id_grouped['filename'].apply(lambda x: x.split('_')[1])
    # Form a unique identifier for each group combining 'file_segment', 'interaction_id', and 'consec_group'
    interaction_id_grouped['unique_consec_group'] = interaction_id_grouped['file_segment'] + '_' + interaction_id_grouped['interaction_id'].astype(str) + '_group' + interaction_id_grouped['consec_group'].astype(str)
    
    # Now, enumerate the frames within each consecutive group
    interaction_id_grouped['interaction_id_consec_frame'] = interaction_id_grouped.groupby('consec_group').cumcount() + 1
    
    return interaction_id_grouped

def calculate_consecutive_frames(df_interactions_file):
    # Sort the DataFrame by 'interaction_id' and 'frame'
    df_sorted = df_interactions_file.sort_values(by=['interaction_id', 'frame'])

    # Group by 'interaction_id'
    interaction_id_grouped = df_sorted.groupby('interaction_id')

    df_with_consec = interaction_id_grouped.apply(identify_consecutive_frames)
    df_with_consec.reset_index(drop=True, inplace=True)
    
    # Drop intermediate columns used for computation
    df_with_consec.drop(columns=['frame_diff', 'is_consecutive', 'consec_group', 'file_segment'], inplace=True)
    
    return df_with_consec

def find_cell_interactions_with_counts(
        t_cells,
        cancer_cells,
        filepath
):
    # Initialize an empty list to store DataFrames from each frame
    all_interactions_data = []
    
    # Extract the start frame number from the filename, assuming a similar naming convention
    start_frame = 1  # Default start frame
    match = re.search(r'start_(\d+)_end_(\d+)', filepath)
    if match:
        start_frame = int(match.group(1))

    for frame_idx in range(t_cells.shape[0]):
        t_cell_frame = t_cells[frame_idx, :, :]
        cancer_cell_frame = cancer_cells[frame_idx, :, :]
        unique_t_cells = np.unique(t_cell_frame[t_cell_frame > 0])
        
        interaction_pairs = []
        for t_cell_id in unique_t_cells:
            t_cell_coords = np.argwhere(t_cell_frame == t_cell_id)
            
            for coord in t_cell_coords:
                x, y = coord
                neighbors = [(i, j) for i in range(max(0, x-1), min(t_cell_frame.shape[0], x+2))
                            for j in range(max(0, y-1), min(t_cell_frame.shape[1], y+2))
                            if (i, j) != (x, y)]
                
                for nx, ny in neighbors:
                    if cancer_cell_frame[nx, ny] > 0:
                        interaction_pairs.append((t_cell_id, cancer_cell_frame[nx, ny]))
        
        # Convert interactions to DataFrame and count duplicates
        df_interactions = pd.DataFrame(interaction_pairs, columns=['t_cell_id', 'cancer_cell_id'])
        df_interactions['contact_pixels'] = 1
        df_interactions = df_interactions.groupby(['t_cell_id', 'cancer_cell_id']).count().reset_index()
        df_interactions['filename'] = os.path.basename(filepath).replace('.zip', '')
        
        # Add the frame number to the DataFrame
        df_interactions['frame'] = start_frame + frame_idx + 1
        
        all_interactions_data.append(df_interactions)
    
    # Concatenate all frame DataFrames into one
    all_interactions_df = pd.concat(all_interactions_data, ignore_index=True)
    
    return all_interactions_df

def classify_contact_pixels(pixels, Q1, Q3):
    if pixels <= Q1:
        return 'minimal'
    elif pixels > Q1 and pixels <= Q3:
        return 'medium'
    else:
        return 'high'

def find_consecutive_interactions(
        df, 
        min_consecutive_frames
):
    # Step 1: Sort by 'filename'
    df_sorted = df.sort_values(by=['filename'])

    # Step 2: Group by 'filename', then sort by 'frame' and 'interaction_id'
    df_sorted = df_sorted.groupby('filename', group_keys=False).apply(
        lambda group: group.sort_values(by=['frame', 'interaction_id'])
    )

    # Step 3: Group by 'filename' and 'interaction_id'
    grouped = df_sorted.groupby(['filename', 'interaction_id'])

    # Step 4: Find and filter consecutive frames within each group
    consecutive_interactions = []
    for name, group in grouped:
        # Detect where the frame number difference is not 1
        non_consecutive = group['frame'].diff() != 1
        
        # Create unique groups for each sequence of consecutive frames
        consec_groups = non_consecutive.cumsum()
        
        # Count the number of frames in each consecutive group
        consec_counts = group.groupby(consec_groups).size()
        
        # Keep only the groups that meet the minimum consecutive frame threshold
        valid_groups = consec_counts[consec_counts >= min_consecutive_frames].index

        # Append valid groups to the list
        for g in valid_groups:
            consecutive_interactions.append(group[consec_groups == g])

    # Return a DataFrame with all valid consecutive interactions
    if consecutive_interactions:
        return pd.concat(consecutive_interactions, ignore_index=True)
    else:
        return pd.DataFrame()

def calculate_average_change(
        group
):
    # Calculate the difference between consecutive rows for both areas without .abs()
    group['t_cell_area_change'] = group['t_cell_area'].diff()
    group['cancer_cell_area_change'] = group['cancer_cell_area'].diff()
    
    # Calculate the average change, ignoring NaN values from .diff() in the first row of each group
    avg_t_cell_area_change = group['t_cell_area_change'].mean()
    avg_cancer_cell_area_change = group['cancer_cell_area_change'].mean()
    
    return pd.Series({
        'avg_t_cell_area_change': avg_t_cell_area_change,
        'avg_cancer_cell_area_change': avg_cancer_cell_area_change
    })

def get_group_from_filename(
        filename, 
        group_logic
):
    for group, prefixes in group_logic.items():
        if any(prefix in filename for prefix in prefixes):
            return group
    return None

def filter_and_label(
        group
):
    # Get the reference frame from frame_x
    ref_frame = group['frame_x'].iloc[0]
    
    # Filter to include only frames within last 10 before frame_x plus the reference frame
    condition = (group['frame_y'] <= ref_frame) & (group['frame_y'] > ref_frame - 11)
    filtered_group = group[condition].copy()
    
    # Sort and label the last 10 frames plus the reference frame
    if not filtered_group.empty:
        filtered_group = filtered_group.sort_values(by='frame_y', ascending=False)
        # Ensure the length of labels matches the number of rows
        filtered_group['interaction_id_consec_frame'] = list(range(0, -len(filtered_group), -1))
    
    return filtered_group

def calculate_cancer_cell_area_change(
        interactions_df, 
        area_df, 
        filename, 
        post_interaction_windows
):
    # Filter the data for interactions with the specified filename
    interactions = interactions_df[interactions_df['filename'] == filename].copy()
    
    # Sort interactions by cancer_cell_id and frame
    interactions.sort_values(by=['cancer_cell_id', 'frame'], inplace=True)
    
    # Initialize dictionary to hold changes for each window
    changes = {window: [] for window in post_interaction_windows}

    # Get the first and last frame numbers in the area_df for bounds checking
    min_frame = area_df['frame'].min()
    max_frame = area_df['frame'].max()
    
    # Group by cancer_cell_id and process interactions for each cancer cell
    for cancer_cell_id, group in interactions.groupby('cancer_cell_id'):
        first_interaction_frame = group['frame'].min()
        last_interaction_frame = group['frame'].max()

        # Find the area just before the first interaction
        pre_interaction_areas = area_df[(area_df['cancer_cell_id'] == cancer_cell_id) &
                                        (area_df['frame'] < first_interaction_frame) &
                                        (area_df['filename'] == filename)]
        
        # If no pre-interaction area data is available, skip to the next cancer cell
        if pre_interaction_areas.empty:
            continue
        
        pre_interaction_area = pre_interaction_areas.nlargest(1, 'frame')['cancer_cell_area'].iloc[0]
        
        # Calculate area change for each window
        for window in post_interaction_windows:
            # Adjust post_interaction_frame if it goes beyond the recorded frames
            post_interaction_frame = min(last_interaction_frame + window, max_frame)
            
            # Get post-interaction areas, up to and including the adjusted post_interaction_frame
            post_interaction_areas = area_df[(area_df['cancer_cell_id'] == cancer_cell_id) &
                                             (area_df['frame'] <= post_interaction_frame) &
                                             (area_df['filename'] == filename)]
            
            # If no post-interaction area data is available, skip to the next window
            if post_interaction_areas.empty:
                continue
            
            # Use the area from the last available frame for the cancer cell
            post_interaction_area = post_interaction_areas.nlargest(1, 'frame')['cancer_cell_area'].iloc[0]
            
            # Calculate the change in area
            area_change = post_interaction_area - pre_interaction_area
            
            # Store the result
            changes[window].append({
                'cancer_cell_id': cancer_cell_id,
                'area_change': area_change,
                'start_frame': first_interaction_frame,
                'end_frame': post_interaction_frame
            })

    # Convert the results to DataFrames
    changes_dfs = {window: pd.DataFrame(data) for window, data in changes.items() if data}

    return changes_dfs

def make_cumulative_area_change_plot(
        test,
        df_consecutive_interactions_dict,
        task_timestamp,
):
    cell_specific_columns = ['t_cell_area', 'cancer_cell_area']

    for cell_specific_column in cell_specific_columns:
        # Define the layout of the figure
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))  # 3 rows and 3 columns
        fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust spacing between subplots

        # Set the overall title for the figure
        fig.suptitle(f'Cumulative Change in {cell_specific_column}', fontsize=16)

        # To store min and max values for y-axis in each row
        row_min_max = []

        # First pass: Calculate the min and max 'area_change' values for each row
        for key, df in df_consecutive_interactions_dict.items():
            df.sort_values(by=['filename', 'interaction_id', 'frame'], inplace=True)
            df['first_area'] = df.groupby(['unique_consec_group'])[cell_specific_column].transform('first')
            df['area_change'] = df[cell_specific_column] - df['first_area']
            
            # Find the min and max 'area_change' in this DataFrame
            min_area_change = df['area_change'].min()
            max_area_change = df['area_change'].max()
            row_min_max.append((min_area_change, max_area_change))

        # Second pass: Plot the data, applying the min and max values per row
        for row_idx, (key, df) in enumerate(df_consecutive_interactions_dict.items()):
            groups = df['group'].unique()
            # Retrieve min and max values for the current row
            ymin, ymax = row_min_max[row_idx]
            
            for col_idx, group in enumerate(groups):
                ax = axes[row_idx, col_idx]
                group_data = df[df['group'] == group]
                unique_consec_groups = group_data['unique_consec_group'].unique()
                
                for unique_group in unique_consec_groups:
                    subgroup_data = group_data[group_data['unique_consec_group'] == unique_group]
                    ax.plot(subgroup_data['interaction_id_consec_frame'], subgroup_data['area_change'], label=f'{unique_group}')
                
                ax.set_ylim([ymin - 5, ymax + 5])  # Set the same y-axis limits for all plots in the row
                if key == 'consec_2':
                    ax.set_title(f'{group}: minimum consecutive frames ≥2')
                if key == 'consec_5':
                    ax.set_title(f'{group}: minimum consecutive frames ≥5') 
                if key == 'consec_10':
                    ax.set_title(f'{group}: minimum consecutive frames ≥10')
                ax.set_xlabel('Consecutive Frames (Normalized Start Frame)')
                ax.set_ylabel('Cumulative Area Change')
                ax.grid(True)
                # ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
        filename_prefix = 't_cell' if cell_specific_column == 't_cell_area' else 'cancer_cell'
        plt.savefig(f'~/Occident-Paper/plots/{filename_prefix}_cumulative_area_change_over_time_{task_timestamp}.pdf')
        plt.close()

def make_mean_area_change_plot(
        test,
        df_consecutive_interactions_dict, 
        task_timestamp,
        mean_area_color_map,
):
    cell_specific_columns = ['t_cell_area', 'cancer_cell_area']

    # Assuming color_map is None or it's equivalent to having no predefined color_map
    for cell_specific_column in cell_specific_columns:
        # Define the layout of the figure for mean area change with standard error
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        fig.suptitle(f'Mean Cumulative Change in {cell_specific_column} with Standard Error', fontsize=16)

        # First pass: Calculate the min and max 'area_change' values for each row
        row_min_max = []
        for key, df in df_consecutive_interactions_dict.items():
            df.sort_values(by=['filename', 'interaction_id', 'frame'], inplace=True)
            df['first_area'] = df.groupby(['interaction_id'])[cell_specific_column].transform('first')
            df['area_change'] = df[cell_specific_column] - df['first_area']

            # Calculate mean and standard error for each 'interaction_id_consec_frame' across all instances within the group
            frame_stats = df.groupby(['group', 'interaction_id_consec_frame'])['area_change'].agg(['mean', 'sem']).reset_index()
            min_val = frame_stats['mean'] - frame_stats['sem']
            max_val = frame_stats['mean'] + frame_stats['sem']
            row_min_max.append((min_val.min(), max_val.max()))

        # Second pass: Plot the mean and standard error data
        for row_idx, (key, df) in enumerate(df_consecutive_interactions_dict.items()):
            groups = df['group'].unique()

            for col_idx, group in enumerate(groups):
                ax = axes[row_idx, col_idx]
                # Only plot frames where the group matches
                frame_stats = df[df['group'] == group].groupby('interaction_id_consec_frame')['area_change'].agg(['mean', 'sem'])

                ax.errorbar(frame_stats.index, frame_stats['mean'], yerr=frame_stats['sem'], fmt='-o', label=f'Group {group}', 
                            color=mean_area_color_map.get(group, 'gray'),  # Use group color or default to gray
                            ecolor='lightgray', elinewidth=2, capsize=0)  # Error bar color
                
                ymin, ymax = row_min_max[row_idx]  # Get min/max for setting y-axis limits
                ax.set_ylim([ymin - 5, ymax + 5])
                if key == 'consec_2':
                    ax.set_title(f'{group}: minimum consecutive frames ≥2')
                if key == 'consec_5':
                    ax.set_title(f'{group}: minimum consecutive frames ≥5') 
                if key == 'consec_10':
                    ax.set_title(f'{group}: minimum consecutive frames ≥10')
                ax.set_xlabel('Consecutive Frames')
                ax.set_ylabel('Mean Cumulative Area Change')
                ax.grid(True)
                # ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename_prefix = 't_cell' if cell_specific_column == 't_cell_area' else 'cancer_cell'
        plt.savefig(f'~/Occident-Paper/plots/{filename_prefix}_mean_cumulative_area_change_over_time_{task_timestamp}.pdf')
        plt.close()


def make_cumulative_area_change_tbls(
        test,
        df_consecutive_interactions_dict,
        task_timestamp,
):
    cell_specific_columns = ['t_cell_area', 'cancer_cell_area']

    final_cumulative_area_change_summary_stats_df = pd.DataFrame()
    for cell_specific_column in cell_specific_columns:
        final_cumulative_area_change_summary_stats_df = pd.DataFrame()
        for i, (key, df) in enumerate(df_consecutive_interactions_dict.items()):
            df.sort_values(by=['filename', 'interaction_id', 'frame'], inplace=True)
            df['first_area'] = df.groupby(['interaction_id'])[cell_specific_column].transform('first')
            df['area_change'] = df[cell_specific_column] - df['first_area']
            cell_groups = df['group'].unique()
            for cell_group in cell_groups:
                cell_data = df[df['group'] == cell_group]
                cumulative_area_change_summary_stats = cell_data['area_change'].describe()
                cumulative_area_change_summary_stats_df = pd.DataFrame(cumulative_area_change_summary_stats)
                cumulative_area_change_summary_stats_df.rename(columns={'area_change': f'{cell_group}_{key}'}, inplace=True)
                final_cumulative_area_change_summary_stats_df = pd.concat([final_cumulative_area_change_summary_stats_df, cumulative_area_change_summary_stats_df], axis=1)
        filename_prefix = 't_cell' if 't_cell_area' in cell_specific_column else 'cancer_cell'
        tbl_filename = f"~/Occident-Paper/tables/{filename_prefix}_cumulative_area_change_summary_stats_{task_timestamp}.csv"
        final_cumulative_area_change_summary_stats_df.to_csv(tbl_filename)
        print(f"{cell_specific_column}:\n{final_cumulative_area_change_summary_stats_df}")
        final_cumulative_area_change_summary_stats_df = pd.DataFrame()

def make_cumulative_area_change_violinplot_subplots(
        test,
        df_consecutive_interactions_dict,
        task_timestamp,
):
    cell_specific_columns = ['t_cell_area', 'cancer_cell_area']

    for cell_specific_column in cell_specific_columns:
        # Define the layout of the figure
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))  # 3 rows and 1 column
        fig.subplots_adjust(hspace=0.5)  # Adjust spacing between subplots

        # Set the overall title for the figure
        fig.suptitle(f'{cell_specific_column} Cumulative Change Violin Plots', fontsize=16)

        # Iterate through each key and its corresponding DataFrame in the dictionary
        for i, (key, df) in enumerate(df_consecutive_interactions_dict.items()):
            df.sort_values(by=['filename', 'interaction_id', 'frame'], inplace=True)
            df['first_area'] = df.groupby(['interaction_id'])[cell_specific_column].transform('first')
            df['area_change'] = df[cell_specific_column] - df['first_area']

            # Select the subplot
            ax = axes[i] if len(df_consecutive_interactions_dict) > 1 else axes

            # Create violin plot
            sns.violinplot(x='group', y='area_change', data=df, ax=ax, scale='width', palette="Set3")
            if key == 'consec_2':
                ax.set_title(f'minimum consecutive frames >2')
            if key == 'consec_5':
                ax.set_title(f'minimum consecutive frames >5') 
            if key == 'consec_10':
                ax.set_title(f'minimum consecutive frames >10')
            ax.set_ylabel('Cumulative Change in Area')
            ax.set_xlabel('Group')

        # Save the figure
        filename_prefix = 't_cell' if 't_cell_area' in cell_specific_column else 'cancer_cell'
        plt.savefig(f'~/Occident-Paper/plots/{filename_prefix}_violin_cumulative_area_change_over_time_{task_timestamp}.pdf')
        plt.close()

def make_roundness_plot_and_table(
        input_df,
        group_logic,
        colors,
        task_timestamp,
        base_save_roundness_table_filename,
        base_save_roundness_plot_filename
):
    roundness_df = input_df[['t_cell_id', 't_cell_roundness', 'filename']].copy()
    roundness_df['group'] = roundness_df['filename'].apply(lambda x: get_group_from_filename(x, group_logic))
    # Group by 'group' and calculate the mean and CI
    grouped = roundness_df.groupby('group')['t_cell_roundness']
    summary_table = grouped.apply(lambda x: mean_confidence_interval(x)).to_dict()
    adjusted_summary_table = {
        'CUL5 KO': summary_table['cul5_ko'],
        'RASA2 KO': summary_table['rasa2_ko'],
        'Safe Harbor KO': summary_table['safe_harbor_ko']
    }
    group_stats = {
        'CUL5 KO': {'mean': summary_table['cul5_ko'][0], 'ci_lower': summary_table['cul5_ko'][1], 'ci_upper': summary_table['cul5_ko'][2], 'n': roundness_df[roundness_df['group'] == 'cul5_ko'].shape[0]},
        'RASA2 KO': {'mean': summary_table['rasa2_ko'][0], 'ci_lower': summary_table['rasa2_ko'][1], 'ci_upper': summary_table['rasa2_ko'][2], 'n': roundness_df[roundness_df['group'] == 'rasa2_ko'].shape[0]},
        'Safe Harbor KO': {'mean': summary_table['safe_harbor_ko'][0], 'ci_lower': summary_table['safe_harbor_ko'][1], 'ci_upper': summary_table['safe_harbor_ko'][2], 'n': roundness_df[roundness_df['group'] == 'safe_harbor_ko'].shape[0]}
    }
    # Calculate SE from CI
    for group, stats_data in group_stats.items():
        stats_data['se'] = (stats_data['ci_upper'] - stats_data['ci_lower']) / (2 * 1.96)
    # Make DataFrame
    df = pd.DataFrame.from_dict(group_stats, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Group'}, inplace=True)
    # Perform pairwise t-tests and calculate z-scores
    results = {}
    z_scores = {}
    groups = list(group_stats.keys())
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1, group2 = groups[i], groups[j]
            stats1, stats2 = group_stats[group1], group_stats[group2]
            t_stat, p_value = ttest_ind_from_stats(
                mean1=stats1['mean'], std1=stats1['se'], nobs1=stats1['n'],
                mean2=stats2['mean'], std2=stats2['se'], nobs2=stats2['n'],
                equal_var=True  # Change this if variances are not assumed to be equal
            )
            # z_score = norm.ppf(1 - p_value/2)  # Two-tailed test
            results[f'p value {group1} vs {group2}'] = p_value
            z_scores[f'z score {group1} vs {group2}'] = t_stat
    
    for test in results.keys():
        df[test] = np.nan  # Initializing the columns with p-values
        z_test = test.replace("p value", "z score")
        df[z_test] = np.nan  # Initializing the columns with z-scores

    # Populate the columns with p-values and z-scores
    for index, row in df.iterrows():
        for test, p_value in results.items():
            groups_in_test = test.split(' vs ')
            if row['Group'] in groups_in_test:
                df.loc[index, test] = p_value
                z_test = test.replace("p value", "z score")
                z_score = z_scores[z_test]
                df.loc[index, z_test] = z_score

    # Save DataFrame to CSV
    save_roundness_table_filename = base_save_roundness_table_filename.format(task_timestamp=task_timestamp)
    df.to_csv(save_roundness_table_filename, index=False)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 10))
    bar_width = 0.35  # Set the width of each bar
    positions = np.arange(len(adjusted_summary_table))  # Bar positions
    for idx, (group, (mean, ci_lower, ci_upper)) in enumerate(adjusted_summary_table.items()):
        color = colors[group]
        ax.bar(idx, mean, color=color, width=bar_width, yerr=[[mean - ci_lower], [ci_upper - mean]], capsize=5, label=group)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Group')
    ax.set_ylabel('Average T Cell Roundness')
    # ax.set_title('Average T Cell Roundness with 95% CI by Group')
    ax.set_xticks(positions)
    ax.set_xticklabels(adjusted_summary_table.keys())
    # plt.legend()
    # Save plot to file
    save_roundness_plot_filename = base_save_roundness_plot_filename.format(task_timestamp=task_timestamp)
    plt.savefig(save_roundness_plot_filename, transparent=True)
    plt.close()

def make_cancer_cell_barplots(
        input_df,
        group_logic,
        colors,
        task_timestamp,
        base_save_cancer_barplots_filename
):
    input_df['group'] = input_df['filename'].apply(lambda x: get_group_from_filename(x, group_logic))
    input_df['group'] = input_df['group'].replace({
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
    })
    for metric in ['cell_area', 'cell_perimeter', 'cell_roundness']:
        print(f"Metric: {metric}")
        filtered_df = input_df[['cell_id', 'frame', metric, 'group']]

        # Group by 'group' and calculate the mean and CI
        grouped = filtered_df.groupby('group')[metric]
        summary_table = grouped.apply(lambda x: mean_confidence_interval(x)).to_dict()
        adjusted_summary_table = {
            'CUL5 KO': summary_table['CUL5 KO'],
            'RASA2 KO': summary_table['RASA2 KO'],
            'Safe Harbor KO': summary_table['Safe Harbor KO']
        }

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 10))
        bar_width = 0.35  # Set the width of each bar
        positions = np.arange(len(adjusted_summary_table))  # Bar positions
        for idx, (group, (mean, ci_lower, ci_upper)) in enumerate(adjusted_summary_table.items()):
            color = colors[group]
            ax.bar(idx, mean, color=color, width=bar_width, yerr=[[mean - ci_lower], [ci_upper - mean]], capsize=5, label=group)
        ax.set_xlabel('Group')
        in_text = metric.replace('_', ' ').title()
        
        # get the base filename from base_save_cancer_barplots_filename
        base_filename = os.path.basename(base_save_cancer_barplots_filename)
        if 'single' in base_filename:
            data_type = 'individual'
        if 'clumped' in base_filename:
            data_type = 'clumped'
        data_type = data_type.title()
        ax.set_ylabel(f'Average {data_type} Cancer {in_text}')
        # ax.set_title(f'Average {data_type} Cancer {in_text} with 95% CI by Group')
        ax.set_xticks(positions)
        ax.set_xticklabels(adjusted_summary_table.keys())

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # plt.legend()
        # Save plot to file
        save_cancer_barplots_filename = base_save_cancer_barplots_filename.format(metric=metric, task_timestamp=task_timestamp)
        plt.savefig(save_cancer_barplots_filename, transparent=True)
        plt.close()


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = np.count_nonzero(~np.isnan(a))  # Count non-NaN entries
    if n == 0:
        return np.nan, np.nan, np.nan  # Return NaN if all are NaNs
    mean = np.nanmean(a)
    se = scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean-h, mean+h

def estimate_se(ci_lower, ci_upper):
    return (ci_upper - ci_lower) / (2 * 1.96)

def compare_groups(
        results_df, 
        base_group_name, 
        subset_comparison_groups
):
    comparisons = []
    total_comparisons = len(subset_comparison_groups) * len(results_df['metric'].unique()) * len(results_df['group'].unique()) # for bonferroni_p correction

    # Filter data for the base group
    base_data = results_df[results_df['frame_group'] == base_group_name]
    
    # Iterate over each comparison group
    for frame_group in subset_comparison_groups:
        comp_data = results_df[results_df['frame_group'] == frame_group]
        
        # Compare for each metric and group
        for metric in base_data['metric'].unique():
            for cell_ko_group in results_df['group'].unique():
                base_subset = base_data[(base_data['group'] == cell_ko_group) & (base_data['metric'] == metric)]
                comp_subset = comp_data[(comp_data['group'] == cell_ko_group) & (comp_data['metric'] == metric)]
                
                if not base_subset.empty and not comp_subset.empty:
                    # Estimate SE for both groups
                    base_se = estimate_se(base_subset['95_ci_lower'].values[0], base_subset['95_ci_upper'].values[0])
                    comp_se = estimate_se(comp_subset['95_ci_lower'].values[0], comp_subset['95_ci_upper'].values[0])
                    
                    # Perform t-test
                    t_stat, p_value = ttest_ind_from_stats(
                        mean1=base_subset['mean'].values[0], std1=base_se, nobs1=base_subset['n'].values[0],
                        mean2=comp_subset['mean'].values[0], std2=comp_se, nobs2=comp_subset['n'].values[0],
                        equal_var=True
                    )
                    
                    # Apply Bonferroni correction
                    bonferroni_p = min(p_value * total_comparisons, 1.0)  # Corrected p-value capped at 1
                    
                    # Save comparison data
                    comparisons.append({
                        'base_group': base_group_name,
                        'comparison_group': frame_group,
                        'metric': metric,
                        'group': cell_ko_group,
                        'base_mean': base_subset['mean'].values[0],
                        'base_lower_ci': base_subset['95_ci_lower'].values[0],
                        'base_upper_ci': base_subset['95_ci_upper'].values[0],
                        'base_se': base_se,
                        'base_n': base_subset['n'].values[0],
                        'comparison_mean': comp_subset['mean'].values[0],
                        'comparison_lower_ci': comp_subset['95_ci_lower'].values[0],
                        'comparison_upper_ci': comp_subset['95_ci_upper'].values[0],
                        'comparison_se': comp_se,
                        'comparison_n': comp_subset['n'].values[0],
                        'p_value': p_value,
                        'z_score': t_stat,
                        'bonferroni_correction': bonferroni_p
                    })
    comparisons = pd.DataFrame(comparisons)
    return comparisons

def make_comparison_tables(
        input_df, 
        task_timestamp, 
        base_filename
):
    for consec_frame_group in input_df['consec_frame_group'].unique():
        df = input_df[input_df['consec_frame_group'] == consec_frame_group]

        df['group'] = df['group'].replace({
            'safe_harbor_ko': 'Safe Harbor KO',
            'rasa2_ko': 'RASA2 KO',
            'cul5_ko': 'CUL5 KO'
        })
        columns_of_interest = [col for col in df.columns if any(kw in col for kw in ['area', 'perimeter', 'roundness', 'velocity'])]
        results = []
        for group in df['group'].unique():
            group_df = df[df['group'] == group]
            # Calculate mean and confidence intervals for frame group -10 to -1
            combined_group = group_df[group_df['interaction_id_consec_frame'].between(-10, -1)]
            for col in columns_of_interest:
                if not combined_group.empty:
                    mean, lower, upper = mean_confidence_interval(combined_group[col])
                    results.append({'frame_group': '-10 to -1', 'group': group, 'metric': col, 'mean': mean, '95_ci_lower': lower, '95_ci_upper': upper, 'n': combined_group.shape[0]})
                else:
                    print("EMPTY!!!!!")
                    results.append({'frame_group': '-10 to -1', 'group': group, 'metric': col, 'mean': None, '95_ci_lower': None, '95_ci_upper': None, 'n': None})
            # Calculate mean and confidence intervals for frame group -10 to 0
            combined_group_10_to_0 = group_df[group_df['interaction_id_consec_frame'].between(-10, 0)]
            for col in columns_of_interest:
                if not combined_group_10_to_0.empty:
                    mean, lower, upper = mean_confidence_interval(combined_group_10_to_0[col])
                    results.append({'frame_group': '-10 to 0', 'group': group, 'metric': col, 'mean': mean, '95_ci_lower': lower, '95_ci_upper': upper, 'n': combined_group_10_to_0.shape[0]})
                else:
                    print("EMPTY!!!!!")
                    results.append({'frame_group': '-10 to 0', 'group': group, 'metric': col, 'mean': None, '95_ci_lower': None, '95_ci_upper': None, 'n': None})
            # Calculate mean and confidence intervals for each group from 0 to 10
            for frame in range(0, 11):  # from 0 to 10
                frame_group_df = group_df[group_df['interaction_id_consec_frame'] == frame]
                for col in columns_of_interest:
                    if not frame_group_df.empty:
                        mean, lower, upper = mean_confidence_interval(frame_group_df[col])
                        results.append({'frame_group': frame, 'group': group, 'metric': col, 'mean': mean, '95_ci_lower': lower, '95_ci_upper': upper, 'n': frame_group_df.shape[0]})
                    else:
                        print("EMPTY!!!!!")
                        results.append({'frame_group': frame, 'group': group, 'metric': col, 'mean': None, '95_ci_lower': None, '95_ci_upper': None, 'n': None})
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        base_groups = ['-10 to -1', '-10 to 0']
        comparison_groups = {
            '-10 to -1': list(range(0, 11)),
            '-10 to 0': list(range(1, 11))
        }
        # Perform comparisons
        all_comparisons = pd.concat([
            compare_groups(results_df=results_df, base_group_name=base_group_name, subset_comparison_groups=comparison_groups[base_group_name]) for base_group_name in base_groups
        ])
        
        all_comparisons.sort_values(['base_group', 'group', 'metric', 'comparison_group'], inplace=True)
        for select_base_group in all_comparisons['base_group'].unique():
            selected_df = all_comparisons[all_comparisons['base_group'] == select_base_group]
            in_text_base_group = select_base_group.replace(' ', '_')
            filename = base_filename.format(consec_frame_group=consec_frame_group, base_group=in_text_base_group, task_timestamp=task_timestamp)
            selected_df.to_csv(filename, index=False)

def calculate_regression_metrics(
        df, 
        x_col, 
        y_cols
):
    regression_results = {}
    for y_col in y_cols:
        regression = linregress(df[x_col], df[y_col])
        slope = regression.slope
        intercept = regression.intercept
        stderr = regression.stderr
        # Calculate 95% CI for the slope
        ci_lower = slope - 1.96 * stderr
        ci_upper = slope + 1.96 * stderr
        
        regression_results[y_col] = {
            'slope': slope,
            'intercept': intercept,
            'se': stderr,
            '95_ci_lower': ci_lower,
            '95_ci_upper': ci_upper
        }
    return regression_results

def make_linear_regression_tables(
        input_df, 
        task_timestamp,
        base_linear_regression_filename
):
    keywords = ['area', 'perimeter', 'roundness'] # 'velocity'
    relevant_columns = [col for col in input_df.columns if any(keyword in col for keyword in keywords)]

    results = []
    for consec_frame_group in input_df['consec_frame_group'].unique():
        df = input_df[input_df['consec_frame_group'] == consec_frame_group]

        # Replace group names as per your specified mappings
        df['group'] = df['group'].replace({
            'safe_harbor_ko': 'Safe Harbor KO',
            'rasa2_ko': 'RASA2 KO',
            'cul5_ko': 'CUL5 KO'
        })

        for group in df['group'].unique():
            group_df = df[df['group'] == group]

            # Define frame ranges
            ranges = {
                '-10 to 0': group_df[group_df['interaction_id_consec_frame'].between(-10, 0)],
                '0 to 10': group_df[group_df['interaction_id_consec_frame'].between(0, 10)],
                '-10 to -1': group_df[group_df['interaction_id_consec_frame'].between(-10, -1)],
                '-1 to 10': group_df[group_df['interaction_id_consec_frame'].between(-1, 10)]
            }

            # Calculate regression metrics for each range
            for range_label, range_df in ranges.items():
                if not range_df.empty:
                    regression_metrics = calculate_regression_metrics(range_df, 'interaction_id_consec_frame', relevant_columns)
                    for metric, metrics_dict in regression_metrics.items():
                        results.append({
                            'consec_frame_group': consec_frame_group,
                            'group': group,
                            'frame_range': range_label,
                            'metric': metric,
                            **metrics_dict  # This unpacks all regression metrics into the results dictionary
                        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.sort_values(['consec_frame_group', 'group', 'metric', 'frame_range'], inplace=True)
    
    # Define two sets of frame ranges for pre-interaction and interaction
    pre_ranges = ['-10 to -1', '-10 to 0']
    inter_ranges = ['-1 to 10', '0 to 10']
    # Filter dataframes for pre-interaction and interaction
    pre_df = results_df[results_df['frame_range'].isin(pre_ranges)].copy()
    inter_df = results_df[results_df['frame_range'].isin(inter_ranges)].copy()
    # Rename columns in pre_df and inter_df to specify their roles
    pre_df.rename({
        'frame_range': 'pre_interaction_frame_range',
        'slope': 'pre_interaction_slope',
        'intercept': 'pre_interaction_intercept',
        'se': 'pre_interaction_se',
        '95_ci_lower': 'pre_interaction_95_ci_lower',
        '95_ci_upper': 'pre_interaction_95_ci_upper'
    }, axis=1, inplace=True)
    inter_df.rename({
        'frame_range': 'interaction_frame_range',
        'slope': 'interaction_slope',
        'intercept': 'interaction_intercept',
        'se': 'interaction_se',
        '95_ci_lower': 'interaction_95_ci_lower',
        '95_ci_upper': 'interaction_95_ci_upper'
    }, axis=1, inplace=True)
    # Map frame ranges to align before and after interaction
    frame_map = {'-10 to -1': '-1 to 10', '-10 to 0': '0 to 10'}
    pre_df['mapped_interaction_frame_range'] = pre_df['pre_interaction_frame_range'].map(frame_map)
    # Merge on the specified keys including the new mapped interaction_frame_range
    merged_df = pd.merge(
        pre_df, 
        inter_df, 
        left_on=['consec_frame_group', 'group', 'metric', 'mapped_interaction_frame_range'],
        right_on=['consec_frame_group', 'group', 'metric', 'interaction_frame_range'],
        how='inner'
    )
    # Select and rearrange columns as needed, you can adjust this part as per your final requirement
    final_columns = [
        'consec_frame_group', 'group', 'metric',
        'pre_interaction_frame_range', 'interaction_frame_range',
        'pre_interaction_slope', 'interaction_slope',
        'pre_interaction_intercept', 'interaction_intercept',
        'pre_interaction_se', 'interaction_se',
        'pre_interaction_95_ci_lower', 'interaction_95_ci_lower',
        'pre_interaction_95_ci_upper', 'interaction_95_ci_upper'
    ]
    final_df = merged_df[final_columns]

    final_df.to_csv(base_linear_regression_filename.format(task_timestamp=task_timestamp), index=False)

def average_metric_barplots(
        input_df, 
        colors, 
        task_timestamp, 
        base_average_metric_filenames
):
    for consec_frame_group in input_df['consec_frame_group'].unique():
        df = input_df[input_df['consec_frame_group'] == consec_frame_group]
        average_metric_filenames = base_average_metric_filenames.format(consec_frame_group=consec_frame_group, task_timestamp=task_timestamp)

        df['group'] = df['group'].replace({
            'safe_harbor_ko': 'Safe Harbor KO',
            'rasa2_ko': 'RASA2 KO',
            'cul5_ko': 'CUL5 KO'
        })
        columns_of_interest = [col for col in df.columns if any(kw in col for kw in ['area', 'perimeter', 'roundness'])]
        results = []
        for group in df['group'].unique():
            group_df = df[df['group'] == group]
            # Calculate mean and confidence intervals for frame group -10 to -1
            combined_group = group_df[group_df['interaction_id_consec_frame'].between(-10, -1)]
            for col in columns_of_interest:
                if not combined_group.empty:
                    mean, lower, upper = mean_confidence_interval(combined_group[col])
                    results.append({'frame_group': '-10 to -1', 'group': group, 'metric': col, 'mean': mean, '95_ci_lower': lower, '95_ci_upper': upper})
                else:
                    results.append({'frame_group': '-10 to -1', 'group': group, 'metric': col, 'mean': None, '95_ci_lower': None, '95_ci_upper': None})
            # Calculate mean and confidence intervals for frame group -10 to 0
            combined_group_10_to_0 = group_df[group_df['interaction_id_consec_frame'].between(-10, 0)]
            for col in columns_of_interest:
                if not combined_group_10_to_0.empty:
                    mean, lower, upper = mean_confidence_interval(combined_group_10_to_0[col])
                    results.append({'frame_group': '-10 to 0', 'group': group, 'metric': col, 'mean': mean, '95_ci_lower': lower, '95_ci_upper': upper})
                else:
                    results.append({'frame_group': '-10 to 0', 'group': group, 'metric': col, 'mean': None, '95_ci_lower': None, '95_ci_upper': None})
            # Calculate mean and confidence intervals for each group from 0 to 10
            for frame in range(0, 11):  # from 0 to 10
                frame_group_df = group_df[group_df['interaction_id_consec_frame'] == frame]
                for col in columns_of_interest:
                    if not frame_group_df.empty:
                        mean, lower, upper = mean_confidence_interval(frame_group_df[col])
                        results.append({'frame_group': frame, 'group': group, 'metric': col, 'mean': mean, '95_ci_lower': lower, '95_ci_upper': upper})
                    else:
                        results.append({'frame_group': frame, 'group': group, 'metric': col, 'mean': None, '95_ci_lower': None, '95_ci_upper': None})
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, metric in zip(axes, columns_of_interest):
            metric_data = results_df[results_df['metric'] == metric].copy()
            bar_plot = sns.barplot(data=metric_data, x='frame_group', y='mean', hue='group', 
                        palette=colors, capsize=.2, ax=ax)
            for bar, idx in zip(bar_plot.patches, range(len(bar_plot.patches))):
                height = bar.get_height()
                x = bar.get_x() + bar.get_width() / 2
                err_lower = metric_data.iloc[idx // 3]['mean'] - metric_data.iloc[idx // 3]['95_ci_lower']
                err_upper = metric_data.iloc[idx // 3]['95_ci_upper'] - metric_data.iloc[idx // 3]['mean']
                ax.errorbar(x, height, yerr=[[err_lower], [err_upper]], fmt='none', capsize=5, color='black', elinewidth=1, capthick=1)
            ax.set_title(f'Mean {metric.replace("_", " ").title()} with 95% Confidence Intervals')
            ax.set_xlabel('Frame Group')
            ax.set_ylabel(f'Mean {metric.replace("_", " ").title()}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.legend(loc='lower right', bbox_to_anchor=(1, 0))
        plt.tight_layout()
        plt.savefig(average_metric_filenames, dpi=300, bbox_inches='tight')
        plt.close()

def individual_average_metric_barplots(
        input_df, 
        colors, 
        task_timestamp, 
        base_average_metric_filenames
):
    for consec_frame_group in input_df['consec_frame_group'].unique():
        df = input_df[input_df['consec_frame_group'] == consec_frame_group]
        average_metric_filenames = base_average_metric_filenames.format(consec_frame_group=consec_frame_group, task_timestamp=task_timestamp)

        df['group'] = df['group'].replace({
            'safe_harbor_ko': 'Safe Harbor KO',
            'rasa2_ko': 'RASA2 KO',
            'cul5_ko': 'CUL5 KO'
        })
        columns_of_interest = [col for col in df.columns if any(kw in col for kw in ['area', 'perimeter', 'roundness'])]
        results = []
        for group in df['group'].unique():
            group_df = df[df['group'] == group]
            # Group for frames -10 to -1
            combined_group = group_df[group_df['interaction_id_consec_frame'].between(-10, -1)]
            for col in columns_of_interest:
                if not combined_group.empty:
                    mean, lower, upper = mean_confidence_interval(combined_group[col])
                    results.append({'frame_group': '-10 to -1', 'group': group, 'metric': col, 'mean': mean, '95_ci_lower': lower, '95_ci_upper': upper})
                else:
                    results.append({'frame_group': '-10 to -1', 'group': group, 'metric': col, 'mean': None, '95_ci_lower': None, '95_ci_upper': None})
            # Calculate mean and confidence intervals for each group from 0 to 10
            for frame in range(0, 11):  # from 0 to 10
                frame_group_df = group_df[group_df['interaction_id_consec_frame'] == frame]
                for col in columns_of_interest:
                    if not frame_group_df.empty:
                        mean, lower, upper = mean_confidence_interval(frame_group_df[col])
                        results.append({'frame_group': frame, 'group': group, 'metric': col, 'mean': mean, '95_ci_lower': lower, '95_ci_upper': upper})
                    else:
                        results.append({'frame_group': frame, 'group': group, 'metric': col, 'mean': None, '95_ci_lower': None, '95_ci_upper': None})
        results_df = pd.DataFrame(results)

        for metric in columns_of_interest:
            fig, ax = plt.subplots(figsize=(8, 6))
            metric_data = results_df[results_df['metric'] == metric].copy()
            # metric_data.sort_values(by=['frame_group', 'group'], inplace=True)  # sorted data -- REMOVE SORT BECAUSE IT RUINS THE X-AXIS ORDER
            bar_plot = sns.barplot(data=metric_data, x='frame_group', y='mean', hue='group', 
                                palette=colors, capsize=.2, ax=ax)
            for bar, idx in zip(bar_plot.patches, range(len(bar_plot.patches))):
                height = bar.get_height()
                x = bar.get_x() + bar.get_width() / 2
                err_lower = metric_data.iloc[idx // 3]['mean'] - metric_data.iloc[idx // 3]['95_ci_lower']
                err_upper = metric_data.iloc[idx // 3]['95_ci_upper'] - metric_data.iloc[idx // 3]['mean']
                ax.errorbar(x, height, yerr=[[err_lower], [err_upper]], fmt='none', capsize=5, color='black', elinewidth=1, capthick=1)
            ax.set_title(f'Mean {metric.replace("_", " ").title()} with 95% Confidence Intervals')
            ax.set_xlabel('Frame Group')
            ax.set_ylabel(f'Mean {metric.replace("_", " ").title()}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.legend(loc='best')
            plt.tight_layout()
            plt.savefig(average_metric_filenames.replace('.pdf', f'_{metric}.pdf'), dpi=300, bbox_inches='tight')
            plt.close()

def make_interaction_analysis_plots(
        df_consecutive_interactions_dict,
        colors,
        task_timestamp,
        base_unique_interaction_table_filename,
        base_interaction_distribution_table_filename,
        base_interaction_distribution_plot_filename,
        base_interaction_distribution_over_time_plot_filename
):
    # Assuming consec_2_df is your input DataFrame already loaded from a dictionary
    consec_2_df = df_consecutive_interactions_dict['consec_2']
    consec_2_df['group'] = consec_2_df['group'].replace({
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
    })
    consec_2_df['file'] = consec_2_df['unique_consec_group'].str.split('_').str[0]

    # making unique interactions table
    idx = consec_2_df.groupby('unique_consec_group')['frame'].idxmin()
    smallest_frame_df = consec_2_df.loc[idx] # Retrieve the rows corresponding to these indices
    smallest_frame_df = smallest_frame_df[['t_cell_id', 'cancer_cell_id', 'unique_consec_group', 'frame', 'interaction_id_max_consec_frames']]
    smallest_frame_df.rename(columns={'frame': 'start_frame'}, inplace=True)
    unique_interaction_table_filename = base_unique_interaction_table_filename.format(task_timestamp=task_timestamp)
    smallest_frame_df.to_csv(unique_interaction_table_filename, index=False)

    interactions_stats_df = pd.DataFrame()
    total_unique_interactions_per_frame_list = []

    # Set up the plot with subplots to get an axis object
    fig, ax = plt.subplots(figsize=(10, 10))
    # Iterate over each unique group
    for group in consec_2_df['group'].unique():
        group_df = consec_2_df[consec_2_df['group'] == group]
        for frame in range(group_df['frame'].min(), group_df['frame'].max() + 1):
            frame_group_df = group_df[group_df['frame'] == frame]
            total_unique_interactions_per_frame = frame_group_df['unique_consec_group'].nunique()
            total_unique_interactions_per_frame_dict = {
                'group': group,
                'frame': frame,
                'total_unique_interactions': total_unique_interactions_per_frame,
            }
            # Collect data in list instead of constantly updating DataFrame
            total_unique_interactions_per_frame_list.append(total_unique_interactions_per_frame_dict)
        # After the loop, summarize unique interactions for the whole group (if needed)
        total_unique_interactions = group_df['unique_consec_group'].nunique()
        max_consec_frame = group_df.drop_duplicates(subset=['unique_consec_group', 'interaction_id_max_consec_frames'], keep='first')['interaction_id_max_consec_frames']
        consec_frame_stats = max_consec_frame.describe()
        data = {'group': group, 'total_unique_interactions': total_unique_interactions}
        data.update(consec_frame_stats.to_dict())

        temp_df = pd.DataFrame([data])
        temp_df.drop(columns=['count', 'std', 'min', '25%', '50%', '75%', 'max'], inplace=True)
        temp_df.rename(columns={'mean': 'mean_duration_interaction_frames'}, inplace=True)
        temp_df['total_unique_t_cells_in_interaction'] = group_df['t_cell_id'].nunique()
        temp_df['total_unique_cancer_cells_in_interaction'] = group_df['cancer_cell_id'].nunique()
        interactions_stats_df = pd.concat([interactions_stats_df, temp_df], ignore_index=True)

        # Plot the distribution of max_consec_frame for the current group
        color = colors.get(group, '#000000')  # Default color is black if group is not found
        max_consec_frame = max_consec_frame[max_consec_frame <= 45]
        ax.hist(max_consec_frame, bins=30, alpha=0.5, color=color, label=f'Group {group}', density=True)  # Changed to density

    # Customize the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Maximum Consecutive Frames')
    ax.set_ylabel('Density')  # Changed label to Density
    ax.legend(title='Group')

    # Save the plot and data
    interaction_distribution_table_filename = base_interaction_distribution_table_filename.format(task_timestamp=task_timestamp)
    interactions_stats_df.to_csv(interaction_distribution_table_filename, index=False)
    total_unique_interactions_per_frame_df = pd.DataFrame(total_unique_interactions_per_frame_list)

    interaction_distribution_plot_filename = base_interaction_distribution_plot_filename.format(task_timestamp=task_timestamp)
    plt.savefig(interaction_distribution_plot_filename, dpi=300, transparent=True)
    plt.close()

    total_interactions_by_file_per_frame_list = []
    for group in consec_2_df['group'].unique():
        group_df = consec_2_df[consec_2_df['group'] == group]
        for frame in range(group_df['frame'].min(), group_df['frame'].max() + 1):
            frame_group_df = group_df[group_df['frame'] == frame]
            counts_by_file = frame_group_df.groupby('file')['unique_consec_group'].nunique()
            # Loop through each file and its count of unique interactions
            for file, count in counts_by_file.items():
                total_interactions_by_file = {
                    'group': group,
                    'frame': frame,
                    'total_unique_interactions': count,  # This is the count for the specific file
                    'file': file,  # This is the file name
                }
                # Collect data in list instead of constantly updating DataFrame
                total_interactions_by_file_per_frame_list.append(total_interactions_by_file)
    # Create DataFrame from list
    total_interactions_by_file_per_frame_df = pd.DataFrame(total_interactions_by_file_per_frame_list)

    # Set up the plot with subplots to get an axis object
    fig, ax = plt.subplots(figsize=(10, 10))
    # sns.set(style="whitegrid")  # Set the style for the plots
    # Prepare data by calculating 'time' from 'frame'
    total_interactions_by_file_per_frame_df['time'] = total_interactions_by_file_per_frame_df['frame'] * 4
    # Plotting each group using a loop to assign custom colors
    for group in total_interactions_by_file_per_frame_df['group'].unique():
        group_data = total_interactions_by_file_per_frame_df[total_interactions_by_file_per_frame_df['group'] == group].copy()  # Make a copy to avoid SettingWithCopyWarning
        # Use seaborn to plot with error bars, plotting 'time' instead of 'frame', using custom colors
        sns.lineplot(ax=ax, x='time', y='total_unique_interactions', data=group_data, label=group, color=colors[group], ci=95)
    # Customize the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (minutes)')  # Adjust label to reflect the new unit
    ax.set_ylabel('Total Unique Interactions')
    ax.legend(title='Group')
    # ax.set_title('Total Unique Interactions Per Time by Group')
    # Save the plot
    interaction_distribution_over_time_plot_filename = base_interaction_distribution_over_time_plot_filename.format(task_timestamp=task_timestamp)
    plt.savefig(interaction_distribution_over_time_plot_filename, dpi=300, transparent=True)
    plt.close()

def plot_perimeter_area_roundness_velocity(
        df, 
        colors, 
        plot_filename
):
    # Standardize group names
    df['group'] = df['group'].replace({
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
    })

    # Determine the prefix based on available columns
    prefix = 't_cell' if 't_cell_area' in df.columns else 'cancer_cell'
    figure_title = 'T Cell' if prefix == 't_cell' else 'Cancer Cell'

    # Extend metrics and titles to include 'Velocity'
    metrics = [f'{prefix}_perimeter', f'{prefix}_area', f'{prefix}_roundness', f'{prefix}_velocity']
    metric_titles = ['Perimeter', 'Area', 'Roundness', 'Velocity']
    consec_frame_groups = ['consec_2', 'consec_5', 'consec_10']
    column_titles = ['≥2 minimum consecutive frames', '≥5 minimum consecutive frames', '≥10 minimum consecutive frames']

    # Create a figure with 4x3 subplots to accommodate the additional metric row
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 20), sharex=True)
    fig.suptitle(f"{figure_title} Metrics Subplots for ≥2, ≥5, and ≥10 Minimum Consecutive Frames", fontsize=16)

    # Initialize row limits for 4 rows
    row_limits = {i: {'min': float('inf'), 'max': float('-inf')} for i in range(4)}

    for col_index, group in enumerate(consec_frame_groups):
        axes[0, col_index].set_title(column_titles[col_index], fontsize=12)
        
        for row_index, metric in enumerate(metrics):
            group_df = df[df['consec_frame_group'] == group]  # Filter the DataFrame for the current consec_frame_group

            ax = axes[row_index, col_index]  # Plot the metric
            sns.lineplot(x='interaction_id_consec_frame', y=metric, hue='group', data=group_df,
                        palette=colors, ax=ax, err_style='band', ci=95)

            # Set plot adjustments
            ax.set_xlabel('Interaction ID Consecutive Frame')
            if col_index == 0:  # Only add y-axis labels to the first column
                ax.set_ylabel(f'Mean {metric_titles[row_index]}')
            else:
                ax.set_ylabel('')

            ax.axvline(x=0, color='gray', linestyle='--')  # Mark the frame 0

            # Adjust legend to show only in the first plot or remove if not necessary
            if row_index == 0 and col_index == 0:
                ax.legend(title='Group')
            else:
                ax.get_legend().remove()

            # Update the min and max for the row if the new values are more extreme
            current_min, current_max = ax.get_ylim()
            if current_min < row_limits[row_index]['min']:
                row_limits[row_index]['min'] = current_min
            if current_max > row_limits[row_index]['max']:
                row_limits[row_index]['max'] = current_max

    # Set the uniform y-limits for each row
    for row_index, ax_row in enumerate(axes):
        for ax in ax_row:
            ax.set_ylim(row_limits[row_index]['min'], row_limits[row_index]['max'])

    # Adjust layout to provide space for titles and plots
    plt.tight_layout(rect=[0.05, 0.03, 0.95, 0.97])
    plt.subplots_adjust(top=0.92)  # Adjust top to make room for the suptitle
    plt.savefig(plot_filename)
    plt.close()

def plot_metrics_individual(
        df, 
        colors, 
        plot_filename_base,  # Base filename to append details to for saving each plot
        timestamp
):
    # Standardize group names
    df['group'] = df['group'].replace({
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
    })

    # Determine the prefix based on available columns
    prefix = 't_cell' if 't_cell_area' in df.columns else 'cancer_cell'
    figure_title_base = 'T Cell' if prefix == 't_cell' else 'Cancer Cell'

    # Metrics setup including 'Velocity'
    metrics = [f'{prefix}_perimeter', f'{prefix}_area', f'{prefix}_roundness', f'{prefix}_velocity']
    metric_titles = ['Perimeter', 'Area', 'Roundness', 'Velocity']
    consec_frame_groups = ['consec_2', 'consec_5', 'consec_10']

    # Descriptive titles for the groups
    group_titles = {
        'consec_2': '≥2 minimum consecutive frames',
        'consec_5': '≥5 minimum consecutive frames',
        'consec_10': '≥10 minimum consecutive frames'
    }

    for col_index, group in enumerate(consec_frame_groups):
        for row_index, metric in enumerate(metrics):
            # Create a new figure for each plot
            fig, ax = plt.subplots(figsize=(10, 10))
            group_df = df[df['consec_frame_group'] == group]  # Filter the DataFrame for the current consec_frame_group

            # Plot the metric
            sns.lineplot(x='interaction_id_consec_frame', y=metric, hue='group', data=group_df,
                        palette=colors, ax=ax, err_style='band', ci=95)

            # Title and labels using the descriptive titles for the group
            descriptive_title = group_titles[group]  # Use the descriptive title for the group
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.set_title(f"{figure_title_base} {metric_titles[row_index]} for {descriptive_title}")
            ax.set_xlabel('Interaction ID Consecutive Frame')
            ax.set_ylabel(f'Mean {metric_titles[row_index]}')

            ax.axvline(x=0, color='gray', linestyle='--')  # Mark the frame 0

            if row_index == 0 and col_index == 0:  # Adjust legend to show only in the first plot
                ax.legend(title='Group')
            else:
                ax.get_legend().remove()

            # Adjust layout to provide space for titles and plots
            plt.tight_layout()

            # Build the filename for each plot
            individual_plot_filename = f"{plot_filename_base}{metric}_{group}_{timestamp}_past50.pdf"
            plt.savefig(individual_plot_filename, transparent=True)
            plt.close()

def plot_t_cell_segmentation_morphology_metrics(
        attached_grouped_results,
        all_grouped_results,
        colors,
        base_attached_t_cell_morphology_over_time_filename,
        base_all_t_cell_morphology_over_time_filename,
        task_timestamp
):
    # Standardize group names
    for df in [attached_grouped_results, all_grouped_results]:
        df['group'] = df['group'].replace({
            'safe_harbor_ko': 'Safe Harbor KO',
            'rasa2_ko': 'RASA2 KO',
            'cul5_ko': 'CUL5 KO'
        })

    # Define the metrics to plot
    metrics_info = {
        'cell_area': 'Area',
        'cell_perimeter': 'Perimeter',
        'cell_roundness': 'Roundness'
    }

    # Create a plot for each metric
    for metric, title in metrics_info.items():
        # Determine y-axis limits based on both datasets to keep y-axis uniform
        min_y = min(
            attached_grouped_results[f'{metric}_mean'].min() - attached_grouped_results[f'{metric}_sem'].max(),
            all_grouped_results[f'{metric}_mean'].min() - all_grouped_results[f'{metric}_sem'].max()
        )
        max_y = max(
            attached_grouped_results[f'{metric}_mean'].max() + attached_grouped_results[f'{metric}_sem'].max(),
            all_grouped_results[f'{metric}_mean'].max() + all_grouped_results[f'{metric}_sem'].max()
        )

        # Plot for attached_grouped_results
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_data(attached_grouped_results, metric, title, colors, ax)
        ax.set_ylim(min_y, max_y)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        attached_filename = base_attached_t_cell_morphology_over_time_filename.format(
            metric=metric, task_timestamp=task_timestamp
        )
        plt.savefig(attached_filename, transparent=True)
        plt.close()

        # Plot for all_grouped_results
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_data(all_grouped_results, metric, title, colors, ax)
        
        # Only set y-limits if the metric is not 'cell_roundness'
        if metric != 'cell_roundness':
            ax.set_ylim(min_y, max_y)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        all_filename = base_all_t_cell_morphology_over_time_filename.format(
            metric=metric, task_timestamp=task_timestamp
        )
        plt.savefig(all_filename, transparent=True)
        plt.close()


def plot_data(df, metric, title, colors, ax):
    for group in df['group'].unique():
        group_data = df[df['group'] == group]
        x = group_data['frame']
        y = group_data[f'{metric}_mean']
        err = group_data[f'{metric}_sem']
        sns.lineplot(x=x, y=y, label=group, color=colors[group], ax=ax)
        ax.fill_between(x, y - err, y + err, color=colors[group], alpha=0.3)
    # ax.set_title(f'{title} by Frame for Different Groups')
    ax.set_xlabel('Frame')
    ax.set_ylabel(f'Mean {title}')
    ax.legend(title='Group')
    # plt.grid(True)
    plt.tight_layout()

def plot_individual_t_cell_segmentation_morphology_metrics(
        attached_grouped_results,
        all_grouped_results,
        individual_colors,
        base_all_and_attached_t_cell_morphology_over_time_filename,
        task_timestamp
):
    # Standardize group names
    group_replacements = {
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
    }
    for df in [attached_grouped_results, all_grouped_results]:
        df['group'] = df['group'].replace(group_replacements)

    # Define the metrics to plot
    metrics_info = {
        'cell_area': 'Area',
        'cell_perimeter': 'Perimeter',
        'cell_roundness': 'Roundness'
    }

    # Create plots for each group and metric
    for metric, title in metrics_info.items():
        for group, group_name in group_replacements.items():
            fig, ax = plt.subplots(figsize=(10, 10))

            # Filter data for the current group
            attached_data = attached_grouped_results[attached_grouped_results['group'] == group_name]
            all_data = all_grouped_results[all_grouped_results['group'] == group_name]

            # Plot data
            plot_data_individual(attached_data, metric, title, individual_colors['Attached T Cells'], 'Attached T Cells', ax)
            plot_data_individual(all_data, metric, title, individual_colors['All T Cells'], 'All T Cells', ax)

            # Set plot details
            # ax.set_title(f'{title} by Frame for {group_name}')
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'Mean {title}')
            ax.legend(title='Group Type')
            # plt.grid(True)
            plt.tight_layout()

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Save the plot
            all_and_attached_t_cell_morphology_over_time_filename = base_all_and_attached_t_cell_morphology_over_time_filename.format(metric=metric, group=group, task_timestamp=task_timestamp)
            plt.savefig(all_and_attached_t_cell_morphology_over_time_filename, transparent=True)
            plt.close()

def plot_individual_and_clumped_cancer_cell_segmentation_morphology_metrics(
        cancer_nuc_grouped_results,
        cancer_samdcl_grouped_results,
        cancer_individual_colors,
        base_single_and_clumped_cancer_cell_morphology_over_time_filename,
        task_timestamp
):
    # Standardize group names
    group_replacements = {
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
    }
    for df in [cancer_nuc_grouped_results, cancer_samdcl_grouped_results]:
        df['group'] = df['group'].replace(group_replacements)

    # Define the metrics to plot
    metrics_info = {
        'cell_area': 'Area',
        'cell_perimeter': 'Perimeter',
        'cell_roundness': 'Roundness'
    }

    # Create plots for each group and metric
    for metric, title in metrics_info.items():
        for group, group_name in group_replacements.items():
            fig, ax = plt.subplots(figsize=(10, 10))

            # Filter data for the current group
            clumped_data = cancer_samdcl_grouped_results[cancer_samdcl_grouped_results['group'] == group_name]
            individual_data = cancer_nuc_grouped_results[cancer_nuc_grouped_results['group'] == group_name]

            # Plot data
            plot_data_individual(clumped_data, metric, title, cancer_individual_colors['Cancer Cell Clumps'], 'Cancer Cell Clumps', ax)
            plot_data_individual(individual_data, metric, title, cancer_individual_colors['Cancer Cell Individual'], 'Cancer Cell Individual', ax)

            # Set plot details
            # ax.set_title(f'{title} by Frame for {group_name}')
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'Mean {title}')
            ax.legend(title='Group Type')
            # plt.grid(True)
            plt.tight_layout()

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Save the plot
            single_and_clumped_cancer_cell_morphology_over_time_filename = base_single_and_clumped_cancer_cell_morphology_over_time_filename.format(metric=metric, group=group, task_timestamp=task_timestamp)
            plt.savefig(single_and_clumped_cancer_cell_morphology_over_time_filename, transparent=True)
            plt.close()

def plot_individual_or_clumped_cancer_cell_segmentation_morphology_and_compute_summary_table_metrics(
        cancer_nuc_grouped_results,
        cancer_samdcl_grouped_results,
        colors,
        base_single_cancer_cell_morphology_over_time_filename,
        base_clumped_cancer_cell_morphology_over_time_filename,
        base_single_cancer_cell_morphology_over_time_table_filename,
        base_clumped_cancer_cell_morphology_over_time_table_filename,
        task_timestamp
):
    # Define the metrics to plot
    metrics_info = {
        'cell_area': 'Area',
        'cell_perimeter': 'Perimeter',
        'cell_roundness': 'Roundness'
    }

    data_dict = {
        'individual': cancer_nuc_grouped_results,
        'clumped': cancer_samdcl_grouped_results
    }

    # Iterate over data types and metrics to create plots and save statistics
    for data_type, df in data_dict.items():
        for metric_prefix, title in metrics_info.items():
            fig, ax = plt.subplots(figsize=(10, 10))
            all_stats = []

            for group_name, color in colors.items():
                # Filter data for the current group
                group_data = df[df['group'] == group_name]

                # Compute 95% CI
                group_data['upper_95_ci'] = group_data[f'{metric_prefix}_mean'] + 1.96 * group_data[f'{metric_prefix}_sem']
                group_data['lower_95_ci'] = group_data[f'{metric_prefix}_mean'] - 1.96 * group_data[f'{metric_prefix}_sem']

                # Save statistics
                summary_stats = group_data.groupby('group').agg(
                    mean=(f'{metric_prefix}_mean', 'mean'),
                    upper_95_ci=('upper_95_ci', 'mean'),
                    lower_95_ci=('lower_95_ci', 'mean'),
                    se=(f'{metric_prefix}_sem', 'mean')
                ).reset_index()
                all_stats.append(summary_stats)

                # Plot data using the original function
                plot_data_individual(group_data, metric_prefix, title, color, f'{group_name} ({data_type.capitalize()})', ax)

            # Consolidate and save statistics to CSV
            final_stats_df = pd.concat(all_stats)
            stats_filename = (base_clumped_cancer_cell_morphology_over_time_table_filename if 'clumped' in data_type 
                                   else base_single_cancer_cell_morphology_over_time_table_filename).format(metric=metric_prefix, task_timestamp=task_timestamp)
            final_stats_df.to_csv(stats_filename, index=False)

            # Set plot details
            # ax.set_title(f'{title} by Frame for All Groups ({data_type.capitalize()})')
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'Mean {title}')
            ax.legend(title='Group Type')
            # plt.grid(True)
            plt.tight_layout()

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Save the plot
            final_save_filename = (base_clumped_cancer_cell_morphology_over_time_filename if 'clumped' in data_type 
                                   else base_single_cancer_cell_morphology_over_time_filename).format(metric=metric_prefix, task_timestamp=task_timestamp)
            plt.savefig(final_save_filename, transparent=True)
            plt.close()

def plot_data_individual(df, metric, title, color, label, ax):
    group_data = df
    x = group_data['frame']
    y = group_data[f'{metric}_mean']
    err = group_data[f'{metric}_sem']
    sns.lineplot(x=x, y=y, label=label, color=color, ax=ax)
    ax.fill_between(x, y - err, y + err, color=color, alpha=0.3)

def plot_cell_area_sum_with_ci(
    cancer_samdcl_df,
    base_clumped_cancer_cell_morphology_sum_over_time_filename,
    task_timestamp,
    colors
):
    data_df = cancer_samdcl_df.copy()
    data_df['group'] = data_df['group'].replace({
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
    })

    # Create 'well' column by extracting the substring between the first and second '_'
    data_df['well'] = data_df['filename'].apply(lambda x: x.split('_')[1])

    # Group by frame, group, and well to sum the cell_area for each well
    area_sum_by_frame_and_well = data_df.groupby(['frame', 'group', 'well'])['cell_area'].sum().reset_index()

    # Group by frame and group to calculate the mean and SEM (standard error of mean)
    area_summary = area_sum_by_frame_and_well.groupby(['frame', 'group']).agg(
        mean_cell_area=('cell_area', 'mean'),
        sem_cell_area=('cell_area', lambda x: x.std() / (len(x) ** 0.5))
    ).reset_index()

    # Compute the 95% confidence intervals
    area_summary['upper_95_ci'] = area_summary['mean_cell_area'] + 1.96 * area_summary['sem_cell_area']
    area_summary['lower_95_ci'] = area_summary['mean_cell_area'] - 1.96 * area_summary['sem_cell_area']

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each group's data with specified colors
    for group_name, color in colors.items():
        group_data = area_summary[area_summary['group'] == group_name]

        # Plot the mean cell area
        ax.plot(group_data['frame'], group_data['mean_cell_area'], label=group_name, color=color)

        # Plot the 95% CI as a shaded area
        ax.fill_between(
            group_data['frame'],
            group_data['lower_95_ci'],
            group_data['upper_95_ci'],
            color=color,
            alpha=0.3
        )
    
    # Set plot details
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean Sum of Cell Area')
    # ax.set_title('Mean Sum of Cell Area by Frame with 95% CI for All Groups')
    ax.legend(title='Group Type', loc='upper left')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Apply tight layout
    plt.tight_layout()

    # Save the plot
    final_save_filename = base_clumped_cancer_cell_morphology_sum_over_time_filename.format(task_timestamp=task_timestamp)
    plt.savefig(final_save_filename, transparent=True)
    plt.close()

def plot_individual_and_clumped_cancer_cell_area_ratio(
        cancer_nuc_grouped_results, 
        cancer_samdcl_grouped_results, 
        colors,
        base_cancer_individual_clumped_area_ratio_over_time_filename, 
        task_timestamp
):
    # Standardize group names
    groups = {
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
    }
    for df in [cancer_nuc_grouped_results, cancer_samdcl_grouped_results]:
        df['group'] = df['group'].replace(groups)

    # Create a single plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Process each group and plot together
    for group, group_name in groups.items():
        # Filter data for the current group
        clumped_data = cancer_samdcl_grouped_results[cancer_samdcl_grouped_results['group'] == group_name]
        individual_data = cancer_nuc_grouped_results[cancer_nuc_grouped_results['group'] == group_name]

        # Calculate ratios and errors
        ratio_data = {
            'frame': clumped_data['frame'].values,
            'ratio_mean': [],
            'ratio_sem': []
        }
        for i in range(len(clumped_data)):
            ratio, error = calculate_ratio_and_error(
                clumped_data['cell_area_mean'].iloc[i], clumped_data['cell_area_sem'].iloc[i],
                individual_data['cell_area_mean'].iloc[i], individual_data['cell_area_sem'].iloc[i]
            )
            ratio_data['ratio_mean'].append(ratio)
            ratio_data['ratio_sem'].append(error)

        # Convert dictionary to DataFrame for easier plotting
        ratio_df = pd.DataFrame(ratio_data)
        plot_ratio_data(ratio_df, group_name, colors[group_name], ax)

    # Set general plot details
    ax.set_title('Area Ratio of Clumped to Individual Cancer Cells Over Time')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Ratio of Areas')
    ax.legend(title='Group')
    # plt.grid(True)
    plt.tight_layout()

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the plot
    cancer_individual_clumped_area_ratio_over_time_filename = base_cancer_individual_clumped_area_ratio_over_time_filename.format(task_timestamp=task_timestamp)
    plt.savefig(cancer_individual_clumped_area_ratio_over_time_filename, transparent=True)
    plt.close()

def calculate_ratio_and_error(numerator_mean, numerator_sem, denominator_mean, denominator_sem):
    """Calculate ratio and propagate error for the ratio of two quantities."""
    ratio = numerator_mean / denominator_mean
    ratio_error = ratio * np.sqrt((numerator_sem / numerator_mean) ** 2 + (denominator_sem / denominator_mean) ** 2)
    return ratio, ratio_error

def plot_ratio_data(group_data, group_name, color, ax):
    """Plot the ratio data using seaborn lineplot and matplotlib fill_between for error shading."""
    x = group_data['frame']
    y = group_data['ratio_mean']
    err = group_data['ratio_sem']
    sns.lineplot(x=x, y=y, label=group_name, color=color, ax=ax)
    ax.fill_between(x, y - err, y + err, color=color, alpha=0.3)

def plot_cancer_cell_segmentation_morphology_metrics(
        cancer_samdcl_grouped_results,
        colors,
        base_cancer_cell_morphology_over_time_filename,
        task_timestamp
):
    cancer_samdcl_grouped_results['group'] = cancer_samdcl_grouped_results['group'].replace({
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
    })

    # Define the metrics to plot
    metrics_info = {
        'cell_area': 'Area',
        'cell_perimeter': 'Perimeter',
        'cell_roundness': 'Roundness'
    }

    # Create a plot for each metric
    for metric, title in metrics_info.items():
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plotting each group with its respective color and error style
        for group in cancer_samdcl_grouped_results['group'].unique():
            group_data = cancer_samdcl_grouped_results[cancer_samdcl_grouped_results['group'] == group]

            # Extracting data for plotting
            x = group_data['frame']
            y = group_data[f'{metric}_mean']
            err = group_data[f'{metric}_sem']

            # Plot line with shaded error band
            sns.lineplot(x=x, y=y, label=group, color=colors[group], ax=ax)
            ax.fill_between(x, y - err, y + err, color=colors[group], alpha=0.3)

        # Setting titles and labels
        # ax.set_title(f'{title} by Frame for Different Groups')
        ax.set_xlabel('Frame')
        ax.set_ylabel(f'Mean {title}')
        ax.legend(title='Group')
        # plt.grid(True)
        plt.tight_layout()

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Save each plot
        cancer_cell_morphology_over_time_filename = base_cancer_cell_morphology_over_time_filename.format(metric=metric, task_timestamp=task_timestamp)
        plt.savefig(cancer_cell_morphology_over_time_filename, transparent=True)
        plt.close()

def run_linear_regression_tests(
        cell_combined_dict,
        task_timestamp,
        base_linear_regression_model_summary_table
):
    
    prefix = 't_cell'
    metrics = [f'{prefix}_perimeter', f'{prefix}_area', f'{prefix}_roundness', f'{prefix}_velocity']

    model_stats_list = []
    for metric in metrics:
        print(f"\nRunning Linear Regression Test for {metric}")

        rasa_2ko = cell_combined_dict['t_cell'][(cell_combined_dict['t_cell']['consec_frame_group'] == 'consec_2') & (cell_combined_dict['t_cell']['group'] == 'rasa2_ko')]
        safe_harbor_ko = cell_combined_dict['t_cell'][(cell_combined_dict['t_cell']['consec_frame_group'] == 'consec_2') & (cell_combined_dict['t_cell']['group'] == 'safe_harbor_ko')]
        cul5_ko = cell_combined_dict['t_cell'][(cell_combined_dict['t_cell']['consec_frame_group'] == 'consec_2') & (cell_combined_dict['t_cell']['group'] == 'cul5_ko')]

        rasa_2ko = rasa_2ko[rasa_2ko[metric].notnull()].reset_index(drop=True)
        safe_harbor_ko = safe_harbor_ko[safe_harbor_ko[metric].notnull()].reset_index(drop=True)
        cul5_ko = cul5_ko[cul5_ko[metric].notnull()].reset_index(drop=True)

        df_is_rasa = pd.DataFrame({'is_rasa': [1] * len(rasa_2ko)})
        df_is_rasa[f'{metric}'] = rasa_2ko[metric]
        df_is_rasa['t'] = rasa_2ko['interaction_id_consec_frame']
        df_is_rasa_sh = pd.DataFrame({'is_rasa': [0] * len(safe_harbor_ko)})
        df_is_rasa_sh[f'{metric}'] = safe_harbor_ko[metric]
        df_is_rasa_sh['t'] = safe_harbor_ko['interaction_id_consec_frame']

        rasa_rasa_cul5 = pd.DataFrame({'is_rasa': [0] * len(cul5_ko)})
        rasa_rasa_cul5[f'{metric}'] = cul5_ko[metric]
        rasa_rasa_cul5['t'] = cul5_ko['interaction_id_consec_frame']

        df_is_sh = pd.DataFrame({'is_sh': [1] * len(safe_harbor_ko)})
        df_is_sh[f'{metric}'] = safe_harbor_ko[metric]
        df_is_sh['t'] = safe_harbor_ko['interaction_id_consec_frame']
        df_is_sh_cul5 = pd.DataFrame({'is_sh': [0] * len(cul5_ko)})
        df_is_sh_cul5[f'{metric}'] = cul5_ko[metric]
        df_is_sh_cul5['t'] = cul5_ko['interaction_id_consec_frame']


        rasa_sh_df = pd.concat([df_is_rasa, df_is_rasa_sh], axis=0).reset_index(drop=True)
        rasa_cul5_df = pd.concat([df_is_rasa, rasa_rasa_cul5], axis=0).reset_index(drop=True)
        sh_cul5_df = pd.concat([df_is_sh, df_is_sh_cul5], axis=0).reset_index(drop=True)

        # Fit the model
        rasa_sh_model = smf.ols(f'{metric} ~ t + is_rasa + is_rasa:t', data=rasa_sh_df).fit()
        rasa_cul5_model = smf.ols(f'{metric} ~ t + is_rasa + is_rasa:t', data=rasa_cul5_df).fit()
        sh_cul5_model = smf.ols(f'{metric} ~ t + is_sh + is_sh:t', data=sh_cul5_df).fit()

        # Get model summary
        rasa_sh_model_summary = rasa_sh_model.summary()
        rasa_cul5_model_summary = rasa_cul5_model.summary()
        sh_cul5_model_summary = sh_cul5_model.summary()

        # Add model summary to the list
        models = [
            ('rasa_sh_model', rasa_sh_model),
            ('rasa_cul5_model', rasa_cul5_model),
            ('sh_cul5_model', sh_cul5_model)
        ]

        for model_name, model in models:
            # Base dict with model name and metric
            model_stats = {
                'model_name': model_name,
                'metric': metric,
                'dep_variable': model.model.endog_names,
                'model_type': type(model.model).__name__,
                'method': 'Least Squares',  # Assuming OLS
                'no_observations': int(model.nobs),
                'df_residuals': int(model.df_resid),
                'df_model': int(model.df_model),
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'prob_f_statistic': model.f_pvalue,
                'log_likelihood': model.llf,
                'aic': model.aic,
                'bic': model.bic
            }
            
            # Extract parameters, standard errors, t-values, p-values, and confidence intervals
            params = model.params
            bse = model.bse
            tvalues = model.tvalues
            pvalues = model.pvalues
            conf_int = model.conf_int()
            conf_int.columns = ['conf_int_low', 'conf_int_high']
            
            # For each parameter, add its statistics
            for param_name in params.index:
                model_stats[f'coef_{param_name}'] = params[param_name]
                model_stats[f'std_err_{param_name}'] = bse[param_name]
                model_stats[f't_value_{param_name}'] = tvalues[param_name]
                model_stats[f'p_value_{param_name}'] = pvalues[param_name]
                model_stats[f'conf_int_low_{param_name}'] = conf_int.loc[param_name, 'conf_int_low']
                model_stats[f'conf_int_high_{param_name}'] = conf_int.loc[param_name, 'conf_int_high']
            
            # Append the model statistics to the list
            model_stats_list.append(model_stats)

    # After the loop over metrics, convert the list to a DataFrame
    model_summary_df = pd.DataFrame(model_stats_list)

    # Save to CSV
    linear_regression_model_summary_table = base_linear_regression_model_summary_table.format(task_timestamp=task_timestamp)
    model_summary_df.to_csv(linear_regression_model_summary_table, index=False)

def run_linear_regression_tests_nontracking_data(
        combined_ratio_df, 
        cancer_nuc_df,
        all_t_cell_df,
        task_timestamp
):
    
    # CLUMPED/SINGLE CELL AREA RATIO
    model_stats_list = []
    metric = "cell_area_ratio"
    print(f"\nRunning Linear Regression Test for {metric}")

    cancer_nuc_df['group'] = cancer_nuc_df['group'].replace({
        'Safe Harbor KO': 'safe_harbor_ko',
        'RASA2 KO': 'rasa2_ko',
        'CUL5 KO': 'cul5_ko'
    })

    rasa_2ko = combined_ratio_df[combined_ratio_df['group'] == 'rasa2_ko']
    safe_harbor_ko = combined_ratio_df[combined_ratio_df['group'] == 'safe_harbor_ko']
    cul5_ko = combined_ratio_df[combined_ratio_df['group'] == 'cul5_ko']

    rasa_2ko = rasa_2ko[rasa_2ko[metric].notnull()].reset_index(drop=True)
    safe_harbor_ko = safe_harbor_ko[safe_harbor_ko[metric].notnull()].reset_index(drop=True)
    cul5_ko = cul5_ko[cul5_ko[metric].notnull()].reset_index(drop=True)

    df_is_rasa = pd.DataFrame({'is_rasa': [1] * len(rasa_2ko)})
    df_is_rasa[f'{metric}'] = rasa_2ko[metric]
    df_is_rasa['t'] = rasa_2ko['frame']
    df_is_rasa_sh = pd.DataFrame({'is_rasa': [0] * len(safe_harbor_ko)})
    df_is_rasa_sh[f'{metric}'] = safe_harbor_ko[metric]
    df_is_rasa_sh['t'] = safe_harbor_ko['frame']

    rasa_rasa_cul5 = pd.DataFrame({'is_rasa': [0] * len(cul5_ko)})
    rasa_rasa_cul5[f'{metric}'] = cul5_ko[metric]
    rasa_rasa_cul5['t'] = cul5_ko['frame']

    df_is_sh = pd.DataFrame({'is_sh': [1] * len(safe_harbor_ko)})
    df_is_sh[f'{metric}'] = safe_harbor_ko[metric]
    df_is_sh['t'] = safe_harbor_ko['frame']
    df_is_sh_cul5 = pd.DataFrame({'is_sh': [0] * len(cul5_ko)})
    df_is_sh_cul5[f'{metric}'] = cul5_ko[metric]
    df_is_sh_cul5['t'] = cul5_ko['frame']


    rasa_sh_df = pd.concat([df_is_rasa, df_is_rasa_sh], axis=0).reset_index(drop=True)
    rasa_cul5_df = pd.concat([df_is_rasa, rasa_rasa_cul5], axis=0).reset_index(drop=True)
    sh_cul5_df = pd.concat([df_is_sh, df_is_sh_cul5], axis=0).reset_index(drop=True)

    # Fit the model
    rasa_sh_model = smf.ols(f'{metric} ~ t + is_rasa + is_rasa:t', data=rasa_sh_df).fit()
    rasa_cul5_model = smf.ols(f'{metric} ~ t + is_rasa + is_rasa:t', data=rasa_cul5_df).fit()
    sh_cul5_model = smf.ols(f'{metric} ~ t + is_sh + is_sh:t', data=sh_cul5_df).fit()

    # Get model summary
    rasa_sh_model_summary = rasa_sh_model.summary()
    rasa_cul5_model_summary = rasa_cul5_model.summary()
    sh_cul5_model_summary = sh_cul5_model.summary()

    # Add model summary to the list
    models = [
        ('rasa_sh_model', rasa_sh_model),
        ('rasa_cul5_model', rasa_cul5_model),
        ('sh_cul5_model', sh_cul5_model)
    ]

    for model_name, model in models:
        # Base dict with model name and metric
        model_stats = {
            'model_name': model_name,
            'metric': metric,
            'dep_variable': model.model.endog_names,
            'model_type': type(model.model).__name__,
            'method': 'Least Squares',  # Assuming OLS
            'no_observations': int(model.nobs),
            'df_residuals': int(model.df_resid),
            'df_model': int(model.df_model),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'prob_f_statistic': model.f_pvalue,
            'log_likelihood': model.llf,
            'aic': model.aic,
            'bic': model.bic
        }
        
        # Extract parameters, standard errors, t-values, p-values, and confidence intervals
        params = model.params
        bse = model.bse
        tvalues = model.tvalues
        pvalues = model.pvalues
        conf_int = model.conf_int()
        conf_int.columns = ['conf_int_low', 'conf_int_high']
        
        # For each parameter, add its statistics
        for param_name in params.index:
            model_stats[f'coef_{param_name}'] = params[param_name]
            model_stats[f'std_err_{param_name}'] = bse[param_name]
            model_stats[f't_value_{param_name}'] = tvalues[param_name]
            model_stats[f'p_value_{param_name}'] = pvalues[param_name]
            model_stats[f'conf_int_low_{param_name}'] = conf_int.loc[param_name, 'conf_int_low']
            model_stats[f'conf_int_high_{param_name}'] = conf_int.loc[param_name, 'conf_int_high']
        
        # Append the model statistics to the list
        model_stats_list.append(model_stats)

    # After the loop over metrics, convert the list to a DataFrame
    model_summary_df = pd.DataFrame(model_stats_list)

    # Save to CSV
    save_path = f"~/Occident-Paper/plots/clumped_to_single_cancer_cell_area_ratio_linear_regression_model_summary_{task_timestamp}.csv"
    save_path = os.path.expanduser(save_path)
    model_summary_df.to_csv(save_path, index=False)



    # ALL T CELL ROUNDNESS
    model_stats_list = []
    metric = "cell_roundness"
    print(f"\nRunning Linear Regression Test for {metric}")

    rasa_2ko = all_t_cell_df[all_t_cell_df['group'] == 'rasa2_ko']
    safe_harbor_ko = all_t_cell_df[all_t_cell_df['group'] == 'safe_harbor_ko']
    cul5_ko = all_t_cell_df[all_t_cell_df['group'] == 'cul5_ko']

    rasa_2ko = rasa_2ko[rasa_2ko[metric].notnull()].reset_index(drop=True)
    safe_harbor_ko = safe_harbor_ko[safe_harbor_ko[metric].notnull()].reset_index(drop=True)
    cul5_ko = cul5_ko[cul5_ko[metric].notnull()].reset_index(drop=True)

    df_is_rasa = pd.DataFrame({'is_rasa': [1] * len(rasa_2ko)})
    df_is_rasa[f'{metric}'] = rasa_2ko[metric]
    df_is_rasa['t'] = rasa_2ko['frame']
    df_is_rasa_sh = pd.DataFrame({'is_rasa': [0] * len(safe_harbor_ko)})
    df_is_rasa_sh[f'{metric}'] = safe_harbor_ko[metric]
    df_is_rasa_sh['t'] = safe_harbor_ko['frame']

    rasa_rasa_cul5 = pd.DataFrame({'is_rasa': [0] * len(cul5_ko)})
    rasa_rasa_cul5[f'{metric}'] = cul5_ko[metric]
    rasa_rasa_cul5['t'] = cul5_ko['frame']

    df_is_sh = pd.DataFrame({'is_sh': [1] * len(safe_harbor_ko)})
    df_is_sh[f'{metric}'] = safe_harbor_ko[metric]
    df_is_sh['t'] = safe_harbor_ko['frame']
    df_is_sh_cul5 = pd.DataFrame({'is_sh': [0] * len(cul5_ko)})
    df_is_sh_cul5[f'{metric}'] = cul5_ko[metric]
    df_is_sh_cul5['t'] = cul5_ko['frame']


    rasa_sh_df = pd.concat([df_is_rasa, df_is_rasa_sh], axis=0).reset_index(drop=True)
    rasa_cul5_df = pd.concat([df_is_rasa, rasa_rasa_cul5], axis=0).reset_index(drop=True)
    sh_cul5_df = pd.concat([df_is_sh, df_is_sh_cul5], axis=0).reset_index(drop=True)

    # Fit the model
    rasa_sh_model = smf.ols(f'{metric} ~ t + is_rasa + is_rasa:t', data=rasa_sh_df).fit()
    rasa_cul5_model = smf.ols(f'{metric} ~ t + is_rasa + is_rasa:t', data=rasa_cul5_df).fit()
    sh_cul5_model = smf.ols(f'{metric} ~ t + is_sh + is_sh:t', data=sh_cul5_df).fit()

    # Get model summary
    rasa_sh_model_summary = rasa_sh_model.summary()
    rasa_cul5_model_summary = rasa_cul5_model.summary()
    sh_cul5_model_summary = sh_cul5_model.summary()

    # Add model summary to the list
    models = [
        ('rasa_sh_model', rasa_sh_model),
        ('rasa_cul5_model', rasa_cul5_model),
        ('sh_cul5_model', sh_cul5_model)
    ]

    for model_name, model in models:
        # Base dict with model name and metric
        model_stats = {
            'model_name': model_name,
            'metric': metric,
            'dep_variable': model.model.endog_names,
            'model_type': type(model.model).__name__,
            'method': 'Least Squares',  # Assuming OLS
            'no_observations': int(model.nobs),
            'df_residuals': int(model.df_resid),
            'df_model': int(model.df_model),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'prob_f_statistic': model.f_pvalue,
            'log_likelihood': model.llf,
            'aic': model.aic,
            'bic': model.bic
        }
        
        # Extract parameters, standard errors, t-values, p-values, and confidence intervals
        params = model.params
        bse = model.bse
        tvalues = model.tvalues
        pvalues = model.pvalues
        conf_int = model.conf_int()
        conf_int.columns = ['conf_int_low', 'conf_int_high']
        
        # For each parameter, add its statistics
        for param_name in params.index:
            model_stats[f'coef_{param_name}'] = params[param_name]
            model_stats[f'std_err_{param_name}'] = bse[param_name]
            model_stats[f't_value_{param_name}'] = tvalues[param_name]
            model_stats[f'p_value_{param_name}'] = pvalues[param_name]
            model_stats[f'conf_int_low_{param_name}'] = conf_int.loc[param_name, 'conf_int_low']
            model_stats[f'conf_int_high_{param_name}'] = conf_int.loc[param_name, 'conf_int_high']
        
        # Append the model statistics to the list
        model_stats_list.append(model_stats)

    # After the loop over metrics, convert the list to a DataFrame
    model_summary_df = pd.DataFrame(model_stats_list)

    # Save to CSV
    save_path = f"~/Occident-Paper/tables/all_t_cell_roundness_linear_regression_model_summary_{task_timestamp}.csv"
    save_path = os.path.expanduser(save_path)
    model_summary_df.to_csv(save_path, index=False)


    # ALL T CELL ROUNDNESS
    model_stats_list = []
    metric = "cell_roundness"
    print(f"\nRunning Linear Regression Test for {metric}")

    rasa_2ko = all_t_cell_df[all_t_cell_df['group'] == 'rasa2_ko']
    safe_harbor_ko = all_t_cell_df[all_t_cell_df['group'] == 'safe_harbor_ko']
    cul5_ko = all_t_cell_df[all_t_cell_df['group'] == 'cul5_ko']

    rasa_2ko = rasa_2ko[rasa_2ko[metric].notnull()].reset_index(drop=True)
    safe_harbor_ko = safe_harbor_ko[safe_harbor_ko[metric].notnull()].reset_index(drop=True)
    cul5_ko = cul5_ko[cul5_ko[metric].notnull()].reset_index(drop=True)

    df_is_rasa = pd.DataFrame({'is_rasa': [1] * len(rasa_2ko)})
    df_is_rasa[f'{metric}'] = rasa_2ko[metric]
    df_is_rasa['t'] = rasa_2ko['frame']
    df_is_rasa_sh = pd.DataFrame({'is_rasa': [0] * len(safe_harbor_ko)})
    df_is_rasa_sh[f'{metric}'] = safe_harbor_ko[metric]
    df_is_rasa_sh['t'] = safe_harbor_ko['frame']

    rasa_rasa_cul5 = pd.DataFrame({'is_rasa': [0] * len(cul5_ko)})
    rasa_rasa_cul5[f'{metric}'] = cul5_ko[metric]
    rasa_rasa_cul5['t'] = cul5_ko['frame']

    df_is_sh = pd.DataFrame({'is_sh': [1] * len(safe_harbor_ko)})
    df_is_sh[f'{metric}'] = safe_harbor_ko[metric]
    df_is_sh['t'] = safe_harbor_ko['frame']
    df_is_sh_cul5 = pd.DataFrame({'is_sh': [0] * len(cul5_ko)})
    df_is_sh_cul5[f'{metric}'] = cul5_ko[metric]
    df_is_sh_cul5['t'] = cul5_ko['frame']


    rasa_sh_df = pd.concat([df_is_rasa, df_is_rasa_sh], axis=0).reset_index(drop=True)
    rasa_cul5_df = pd.concat([df_is_rasa, rasa_rasa_cul5], axis=0).reset_index(drop=True)
    sh_cul5_df = pd.concat([df_is_sh, df_is_sh_cul5], axis=0).reset_index(drop=True)

    # Fit the model
    rasa_sh_model = smf.ols(f'{metric} ~ t + is_rasa + is_rasa:t', data=rasa_sh_df).fit()
    rasa_cul5_model = smf.ols(f'{metric} ~ t + is_rasa + is_rasa:t', data=rasa_cul5_df).fit()
    sh_cul5_model = smf.ols(f'{metric} ~ t + is_sh + is_sh:t', data=sh_cul5_df).fit()

    # Get model summary
    rasa_sh_model_summary = rasa_sh_model.summary()
    rasa_cul5_model_summary = rasa_cul5_model.summary()
    sh_cul5_model_summary = sh_cul5_model.summary()

    # Add model summary to the list
    models = [
        ('rasa_sh_model', rasa_sh_model),
        ('rasa_cul5_model', rasa_cul5_model),
        ('sh_cul5_model', sh_cul5_model)
    ]

    for model_name, model in models:
        # Base dict with model name and metric
        model_stats = {
            'model_name': model_name,
            'metric': metric,
            'dep_variable': model.model.endog_names,
            'model_type': type(model.model).__name__,
            'method': 'Least Squares',  # Assuming OLS
            'no_observations': int(model.nobs),
            'df_residuals': int(model.df_resid),
            'df_model': int(model.df_model),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'prob_f_statistic': model.f_pvalue,
            'log_likelihood': model.llf,
            'aic': model.aic,
            'bic': model.bic
        }
        
        # Extract parameters, standard errors, t-values, p-values, and confidence intervals
        params = model.params
        bse = model.bse
        tvalues = model.tvalues
        pvalues = model.pvalues
        conf_int = model.conf_int()
        conf_int.columns = ['conf_int_low', 'conf_int_high']
        
        # For each parameter, add its statistics
        for param_name in params.index:
            model_stats[f'coef_{param_name}'] = params[param_name]
            model_stats[f'std_err_{param_name}'] = bse[param_name]
            model_stats[f't_value_{param_name}'] = tvalues[param_name]
            model_stats[f'p_value_{param_name}'] = pvalues[param_name]
            model_stats[f'conf_int_low_{param_name}'] = conf_int.loc[param_name, 'conf_int_low']
            model_stats[f'conf_int_high_{param_name}'] = conf_int.loc[param_name, 'conf_int_high']
        
        # Append the model statistics to the list
        model_stats_list.append(model_stats)

    # After the loop over metrics, convert the list to a DataFrame
    model_summary_df = pd.DataFrame(model_stats_list)

    # Save to CSV
    save_path = f"~/Occident-Paper/tables/all_t_cell_roundness_linear_regression_model_summary_{task_timestamp}.csv"
    save_path = os.path.expanduser(save_path)
    model_summary_df.to_csv(save_path, index=False)

    # SINGLE CANCER CELL ROUNDNESS
    model_stats_list = []
    metric = "cell_roundness"
    print(f"\nRunning Linear Regression Test for {metric}")

    rasa_2ko = cancer_nuc_df[cancer_nuc_df['group'] == 'rasa2_ko']
    safe_harbor_ko = cancer_nuc_df[cancer_nuc_df['group'] == 'safe_harbor_ko']
    cul5_ko = cancer_nuc_df[cancer_nuc_df['group'] == 'cul5_ko']

    rasa_2ko = rasa_2ko[rasa_2ko[metric].notnull()].reset_index(drop=True)
    safe_harbor_ko = safe_harbor_ko[safe_harbor_ko[metric].notnull()].reset_index(drop=True)
    cul5_ko = cul5_ko[cul5_ko[metric].notnull()].reset_index(drop=True)

    df_is_rasa = pd.DataFrame({'is_rasa': [1] * len(rasa_2ko)})
    df_is_rasa[f'{metric}'] = rasa_2ko[metric]
    df_is_rasa['t'] = rasa_2ko['frame']
    df_is_rasa_sh = pd.DataFrame({'is_rasa': [0] * len(safe_harbor_ko)})
    df_is_rasa_sh[f'{metric}'] = safe_harbor_ko[metric]
    df_is_rasa_sh['t'] = safe_harbor_ko['frame']

    rasa_rasa_cul5 = pd.DataFrame({'is_rasa': [0] * len(cul5_ko)})
    rasa_rasa_cul5[f'{metric}'] = cul5_ko[metric]
    rasa_rasa_cul5['t'] = cul5_ko['frame']

    df_is_sh = pd.DataFrame({'is_sh': [1] * len(safe_harbor_ko)})
    df_is_sh[f'{metric}'] = safe_harbor_ko[metric]
    df_is_sh['t'] = safe_harbor_ko['frame']
    df_is_sh_cul5 = pd.DataFrame({'is_sh': [0] * len(cul5_ko)})
    df_is_sh_cul5[f'{metric}'] = cul5_ko[metric]
    df_is_sh_cul5['t'] = cul5_ko['frame']


    rasa_sh_df = pd.concat([df_is_rasa, df_is_rasa_sh], axis=0).reset_index(drop=True)
    rasa_cul5_df = pd.concat([df_is_rasa, rasa_rasa_cul5], axis=0).reset_index(drop=True)
    sh_cul5_df = pd.concat([df_is_sh, df_is_sh_cul5], axis=0).reset_index(drop=True)

    # Fit the model
    rasa_sh_model = smf.ols(f'{metric} ~ t + is_rasa + is_rasa:t', data=rasa_sh_df).fit()
    rasa_cul5_model = smf.ols(f'{metric} ~ t + is_rasa + is_rasa:t', data=rasa_cul5_df).fit()
    sh_cul5_model = smf.ols(f'{metric} ~ t + is_sh + is_sh:t', data=sh_cul5_df).fit()

    # Get model summary
    rasa_sh_model_summary = rasa_sh_model.summary()
    rasa_cul5_model_summary = rasa_cul5_model.summary()
    sh_cul5_model_summary = sh_cul5_model.summary()

    # Add model summary to the list
    models = [
        ('rasa_sh_model', rasa_sh_model),
        ('rasa_cul5_model', rasa_cul5_model),
        ('sh_cul5_model', sh_cul5_model)
    ]

    for model_name, model in models:
        # Base dict with model name and metric
        model_stats = {
            'model_name': model_name,
            'metric': metric,
            'dep_variable': model.model.endog_names,
            'model_type': type(model.model).__name__,
            'method': 'Least Squares',  # Assuming OLS
            'no_observations': int(model.nobs),
            'df_residuals': int(model.df_resid),
            'df_model': int(model.df_model),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'prob_f_statistic': model.f_pvalue,
            'log_likelihood': model.llf,
            'aic': model.aic,
            'bic': model.bic
        }
        
        # Extract parameters, standard errors, t-values, p-values, and confidence intervals
        params = model.params
        bse = model.bse
        tvalues = model.tvalues
        pvalues = model.pvalues
        conf_int = model.conf_int()
        conf_int.columns = ['conf_int_low', 'conf_int_high']
        
        # For each parameter, add its statistics
        for param_name in params.index:
            model_stats[f'coef_{param_name}'] = params[param_name]
            model_stats[f'std_err_{param_name}'] = bse[param_name]
            model_stats[f't_value_{param_name}'] = tvalues[param_name]
            model_stats[f'p_value_{param_name}'] = pvalues[param_name]
            model_stats[f'conf_int_low_{param_name}'] = conf_int.loc[param_name, 'conf_int_low']
            model_stats[f'conf_int_high_{param_name}'] = conf_int.loc[param_name, 'conf_int_high']
        
        # Append the model statistics to the list
        model_stats_list.append(model_stats)

    # After the loop over metrics, convert the list to a DataFrame
    model_summary_df = pd.DataFrame(model_stats_list)

    # Save to CSV
    save_path = f"~/Occident-Paper/tables/single_cancer_cell_roundness_linear_regression_model_summary_{task_timestamp}.csv"
    save_path = os.path.expanduser(save_path)
    model_summary_df.to_csv(save_path, index=False)