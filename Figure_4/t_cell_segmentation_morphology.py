import sys
import os
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats
from occident.utils import load_deepcell_object, mean_confidence_interval, estimate_se
from occident.tracking import (
    process_all_frames,
    get_group_from_filename
)

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

sys.path.append(os.path.expanduser('.'))
from cell_tracking_helper_functions import(
    make_cancer_cell_barplots,
    plot_t_cell_segmentation_morphology_metrics,
    plot_individual_t_cell_segmentation_morphology_metrics,
    plot_individual_and_clumped_cancer_cell_segmentation_morphology_metrics,
    plot_individual_or_clumped_cancer_cell_segmentation_morphology_and_compute_summary_table_metrics,
    plot_cell_area_sum_with_ci,
    plot_individual_and_clumped_cancer_cell_area_ratio,
    plot_cancer_cell_segmentation_morphology_metrics,
    run_linear_regression_tests_nontracking_data,
    calculate_ratio_and_error
)

def cell_segmentation_morphology(
        test,
        directory_path,
        group_logic,
        colors,
        individual_colors,
):
    task_timestamp = datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d_%H:%M:%S')
    base_save_single_cancer_barplots_filename = './plots/{metric}_single_cancer_cell_morphology_over_time_barplot_{task_timestamp}.pdf'
    base_save_clumped_cancer_barplots_filename = './plots/{metric}_clumped_cancer_cell_morphology_over_time_barplot_{task_timestamp}.pdf'
    base_all_and_attached_t_cell_morphology_over_time_filename = './plots/{metric}_{group}_all_and_attached_t_cell_morphology_over_time_{task_timestamp}.pdf'
    base_attached_t_cell_morphology_over_time_filename = './plots/{metric}_attached_t_cell_morphology_over_time_{task_timestamp}.pdf'
    base_all_t_cell_morphology_over_time_filename = './plots/{metric}_all_t_cell_morphology_over_time_{task_timestamp}.pdf'
    base_single_and_clumped_cancer_cell_morphology_over_time_filename = './plots/{metric}_{group}_single_and_clumped_cancer_cell_morphology_over_time_{task_timestamp}.pdf'
    base_single_cancer_cell_morphology_over_time_filename = './plots/{metric}_single_cancer_cell_morphology_over_time_{task_timestamp}.pdf'
    base_clumped_cancer_cell_morphology_over_time_filename = './plots/{metric}_clumped_cancer_cell_morphology_over_time_{task_timestamp}.pdf'
    base_single_cancer_cell_morphology_over_time_table_filename = './tables/{metric}_single_cancer_cell_morphology_over_time_{task_timestamp}.csv'
    base_clumped_cancer_cell_morphology_over_time_table_filename = './tables/{metric}_clumped_cancer_cell_morphology_over_time_{task_timestamp}.csv'
    base_cancer_cell_morphology_over_time_filename = './plots/{metric}_cancer_cell_morphology_over_time_{task_timestamp}.pdf'
    base_cancer_individual_clumped_area_ratio_over_time_filename = './plots/single_and_clumped_cancer_cell_area_ratio_over_time_{task_timestamp}.pdf'
    base_clumped_cancer_cell_morphology_sum_over_time_filename = "./plots/cell_area_sum_clumped_cancer_cell_morphology_over_time_{task_timestamp}.pdf"
    
    base_save_single_cancer_barplots_filename = os.path.expanduser(base_save_single_cancer_barplots_filename)
    base_save_clumped_cancer_barplots_filename = os.path.expanduser(base_save_clumped_cancer_barplots_filename)
    base_all_and_attached_t_cell_morphology_over_time_filename = os.path.expanduser(base_all_and_attached_t_cell_morphology_over_time_filename)
    base_attached_t_cell_morphology_over_time_filename = os.path.expanduser(base_attached_t_cell_morphology_over_time_filename)
    base_all_t_cell_morphology_over_time_filename = os.path.expanduser(base_all_t_cell_morphology_over_time_filename)
    base_single_and_clumped_cancer_cell_morphology_over_time_filename = os.path.expanduser(base_single_and_clumped_cancer_cell_morphology_over_time_filename)
    base_single_cancer_cell_morphology_over_time_filename = os.path.expanduser(base_single_cancer_cell_morphology_over_time_filename)
    base_clumped_cancer_cell_morphology_over_time_filename = os.path.expanduser(base_clumped_cancer_cell_morphology_over_time_filename)
    base_single_cancer_cell_morphology_over_time_table_filename = os.path.expanduser(base_single_cancer_cell_morphology_over_time_table_filename)
    base_clumped_cancer_cell_morphology_over_time_table_filename = os.path.expanduser(base_clumped_cancer_cell_morphology_over_time_table_filename)
    base_cancer_cell_morphology_over_time_filename = os.path.expanduser(base_cancer_cell_morphology_over_time_filename)
    base_cancer_individual_clumped_area_ratio_over_time_filename = os.path.expanduser(base_cancer_individual_clumped_area_ratio_over_time_filename)
    base_clumped_cancer_cell_morphology_sum_over_time_filename = os.path.expanduser(base_clumped_cancer_cell_morphology_sum_over_time_filename)
    
    directory_path = os.path.expanduser(directory_path)
    filepaths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    codes = [code for sublist in group_logic.values() for code in sublist]
    filtered_filepaths = [filepath for filepath in filepaths if any(code in os.path.basename(filepath) for code in codes)]
    filtered_filepaths = [path for path in filtered_filepaths if 'start_0_end_50' not in path]
    if test:
        print(f"***Loading test env***")
        filtered_filepaths = [filepath for filepath in filtered_filepaths if any(code in os.path.basename(filepath) for code in [
            'B3',
            'B7',
            'E3'
            ])]
    
    all_t_cell_mask_dict = {}
    all_attached_t_cells_dict = {}
    cancer_nuc_mask_dict = {}
    all_cancer_samdcl_masks_dict = {}
    total_iterations = len(filtered_filepaths)
    for current_iteration, filepath in enumerate(filtered_filepaths, start=1):
        filename = os.path.basename(filepath)
        print(f"\nIteration {current_iteration}/{total_iterations}")
        print(f"Processing data for file: {filename}")
        
        print(f"Loading data")
        dcl_ob = load_deepcell_object(filepath)

        dcl_y = dcl_ob['y'][:,:,:,0,:]
        cancer_nuc_mask = dcl_y[0,:,:,:]
        all_t_cell_mask = dcl_y[1,:,:,:]
        all_attached_t_cells = dcl_y[4,:,:,:]
        all_cancer_samdcl_masks = dcl_y[3,:,:,:]

        print(f"Processing 'all_tcell_mask'")
        all_t_cell_mask_df = process_all_frames(all_t_cell_mask, filename)
        all_t_cell_mask_df['cell_roundness'] = 4 * np.pi * all_t_cell_mask_df['cell_area'] / (all_t_cell_mask_df['cell_perimeter'] ** 2)
        all_t_cell_mask_dict[filename] = all_t_cell_mask_df
        print(f"Processing 'all_attached_tcells'")
        all_attached_t_cells_df = process_all_frames(all_attached_t_cells, filename, use_connected_component_labeling = True)
        all_attached_t_cells_df['cell_roundness'] = 4 * np.pi * all_attached_t_cells_df['cell_area'] / (all_attached_t_cells_df['cell_perimeter'] ** 2)
        all_attached_t_cells_dict[filename] = all_attached_t_cells_df
        print(f"Processing 'cancer_nuc_mask")
        cancer_nuc_mask_df = process_all_frames(cancer_nuc_mask, filename)
        cancer_nuc_mask_df['cell_roundness'] = 4 * np.pi * cancer_nuc_mask_df['cell_area'] / (cancer_nuc_mask_df['cell_perimeter'] ** 2)
        cancer_nuc_mask_dict[filename] = cancer_nuc_mask_df
        print(f"Processing 'all_cancer_samdcl_masks'")
        all_cancer_samdcl_masks_df = process_all_frames(all_cancer_samdcl_masks, filename)
        all_cancer_samdcl_masks_df['cell_roundness'] = 4 * np.pi * all_cancer_samdcl_masks_df['cell_area'] / (all_cancer_samdcl_masks_df['cell_perimeter'] ** 2)
        all_cancer_samdcl_masks_dict[filename] = all_cancer_samdcl_masks_df
    
    print(f"\nCombining DataFrames")
    all_t_cell_df = pd.concat(all_t_cell_mask_dict, ignore_index=True)
    attached_t_cell_df = pd.concat(all_attached_t_cells_dict, ignore_index=True)
    cancer_nuc_df = pd.concat(cancer_nuc_mask_dict, ignore_index=True)
    cancer_samdcl_df = pd.concat(all_cancer_samdcl_masks_dict, ignore_index=True)

    all_t_cell_df['group'] = all_t_cell_df['filename'].apply(lambda x: get_group_from_filename(x, group_logic))
    attached_t_cell_df['group'] = attached_t_cell_df['filename'].apply(lambda x: get_group_from_filename(x, group_logic))
    cancer_nuc_df['group'] = cancer_nuc_df['filename'].apply(lambda x: get_group_from_filename(x, group_logic))
    cancer_samdcl_df['group'] = cancer_samdcl_df['filename'].apply(lambda x: get_group_from_filename(x, group_logic))

    print(f"\nPerforming All T Cell and Attached T Cell Analysis")
    columns_of_interest = [col for col in all_t_cell_df.columns if any(kw in col for kw in ['cell_area', 'cell_perimeter', 'cell_roundness'])]
    all_t_cell_results = []
    attached_t_cell_results = []
    p_values = []
    t_statistics = []
    for group in all_t_cell_df['group'].unique():
        all_t_cell_select_df = all_t_cell_df[all_t_cell_df['group'] == group]
        attached_t_cell_select_df = attached_t_cell_df[attached_t_cell_df['group'] == group]

        for col in columns_of_interest:
            all_t_cell_mean, all_t_cell_lower, all_t_cell_upper = mean_confidence_interval(all_t_cell_select_df[col])
            all_t_cell_se = estimate_se(all_t_cell_lower, all_t_cell_upper)
            all_t_cell_results.append({'metric': col, 'group': group, 'all_t_cell_mean': all_t_cell_mean, 'all_t_cell_95_ci_lower': all_t_cell_lower, 'all_t_cell_95_ci_upper': all_t_cell_upper, 'all_t_cell_se': all_t_cell_se, 'all_t_cell_n': all_t_cell_select_df[col].shape[0]})

            attached_t_cell_mean, attached_t_cell_lower, attached_t_cell_upper = mean_confidence_interval(attached_t_cell_select_df[col])
            attached_t_cell_se = estimate_se(attached_t_cell_lower, attached_t_cell_upper)
            attached_t_cell_results.append({'metric': col, 'group': group, 'attached_t_cell_mean': attached_t_cell_mean, 'attached_t_cell_95_ci_lower': attached_t_cell_lower, 'attached_t_cell_95_ci_upper': attached_t_cell_upper, 'attached_t_cell_se': attached_t_cell_se, 'attached_t_cell_n': attached_t_cell_select_df[col].shape[0]})

            # Perform t-test
            t_stat, p_value = ttest_ind_from_stats(
                mean1=all_t_cell_mean, std1=all_t_cell_se, nobs1=all_t_cell_select_df[col].shape[0],
                mean2=attached_t_cell_mean, std2=attached_t_cell_se, nobs2=attached_t_cell_select_df[col].shape[0],
                equal_var=True
            )
            p_values.append(p_value)
            t_statistics.append(t_stat)
    for result, p_value, t_stat in zip(attached_t_cell_results, p_values, t_statistics):
        result['p_value'] = p_value
        result['z_score'] = t_stat
    all_t_cell_results_df = pd.DataFrame(all_t_cell_results)
    attached_t_cell_results_df = pd.DataFrame(attached_t_cell_results)
    final_df = all_t_cell_results_df.merge(
        attached_t_cell_results_df, 
        on=['metric', 'group'], 
        how='left', 
        suffixes=('_all', '_attached')
    )
    save_filename = f'./tables/all_t_cell_and_attached_t_cell_morphology_{task_timestamp}.csv'
    save_filename = os.path.expanduser(save_filename)
    final_df.to_csv(save_filename, index=False)
    
    print(f"\nPerforming Single Cancer Cell and Clumped Cancer Cell Analysis")
    columns_of_interest = [col for col in cancer_nuc_df.columns if any(kw in col for kw in ['cell_area', 'cell_perimeter', 'cell_roundness'])]
    single_cancer_results = []
    clumped_cancer_results = []
    p_values = []
    t_statistics = []
    for group in cancer_nuc_df['group'].unique():
        single_cancer_cells_select_df = cancer_nuc_df[cancer_nuc_df['group'] == group]
        clumped_cancer_cells_select_df = cancer_samdcl_df[cancer_samdcl_df['group'] == group]

        for col in columns_of_interest:
            single_cancer_cell_mean, single_cancer_cell_lower, single_cancer_cell_upper = mean_confidence_interval(single_cancer_cells_select_df[col])
            single_cancer_cell_se = estimate_se(single_cancer_cell_lower, single_cancer_cell_upper)
            single_cancer_results.append({'metric': col, 'group': group, 'single_cancer_cell_mean': single_cancer_cell_mean, 'single_cancer_cell_95_ci_lower': single_cancer_cell_lower, 'single_cancer_cell_95_ci_upper': single_cancer_cell_upper, 'single_cancer_cell_se': single_cancer_cell_se, 'single_cancer_cell_n': single_cancer_cells_select_df[col].shape[0]})

            clumped_cancer_cell_mean, clumped_cancer_cell_lower, clumped_cancer_cell_upper = mean_confidence_interval(clumped_cancer_cells_select_df[col])
            clumped_cancer_cell_se = estimate_se(clumped_cancer_cell_lower, clumped_cancer_cell_upper)
            clumped_cancer_results.append({'metric': col, 'group': group, 'clumped_cancer_cell_mean': clumped_cancer_cell_mean, 'clumped_cancer_cell_95_ci_lower': clumped_cancer_cell_lower, 'clumped_cancer_cell_95_ci_upper': clumped_cancer_cell_upper, 'clumped_cancer_cell_se': clumped_cancer_cell_se, 'clumped_cancer_cell_n': clumped_cancer_cells_select_df[col].shape[0]})

            # Perform t-test
            t_stat, p_value = ttest_ind_from_stats(
                mean1=single_cancer_cell_mean, std1=single_cancer_cell_se, nobs1=single_cancer_cells_select_df[col].shape[0],
                mean2=clumped_cancer_cell_mean, std2=clumped_cancer_cell_se, nobs2=clumped_cancer_cells_select_df[col].shape[0],
                equal_var=True
            )
            p_values.append(p_value)
            t_statistics.append(t_stat)
    for result, p_value, t_stat in zip(clumped_cancer_results, p_values, t_statistics):
        result['p_value'] = p_value
        result['z_score'] = t_stat
    single_cancer_results_df = pd.DataFrame(single_cancer_results)
    clumped_cancer_results_df = pd.DataFrame(clumped_cancer_results)
    final_df = single_cancer_results_df.merge(
        clumped_cancer_results_df, 
        on=['metric', 'group'], 
        how='left', 
        suffixes=('_single', '_clumped')
    )
    save_filename = f'./tables/single_and_clumped_cancer_cell_morphology_{task_timestamp}.csv'
    save_filename = os.path.expanduser(save_filename)
    final_df.to_csv(save_filename, index=False)

    print(f"\nPlotting Morphology Over Time")
    attached_grouped_results = attached_t_cell_df.groupby(['group', 'frame']).agg({
        'cell_area': ['mean', 'sem'],  # sem computes the standard error of the mean
        'cell_perimeter': ['mean', 'sem'],
        'cell_roundness': ['mean', 'sem']
    })
    attached_grouped_results.columns = ['_'.join(col).strip() for col in attached_grouped_results.columns.values]
    attached_grouped_results.reset_index(inplace=True)
    attached_grouped_results['group'] = attached_grouped_results['group'].replace({
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
        })
    
    all_grouped_results = all_t_cell_df.groupby(['group', 'frame']).agg({
        'cell_area': ['mean', 'sem'],  # sem computes the standard error of the mean
        'cell_perimeter': ['mean', 'sem'],
        'cell_roundness': ['mean', 'sem']
    })
    all_grouped_results.columns = ['_'.join(col).strip() for col in all_grouped_results.columns.values]
    all_grouped_results.reset_index(inplace=True)
    all_grouped_results['group'] = all_grouped_results['group'].replace({
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
        })
    
    cancer_nuc_grouped_results = cancer_nuc_df.groupby(['group', 'frame']).agg({
        'cell_area': ['mean', 'sem'],  # sem computes the standard error of the mean
        'cell_perimeter': ['mean', 'sem'],
        'cell_roundness': ['mean', 'sem']
    })
    cancer_nuc_grouped_results.columns = ['_'.join(col).strip() for col in cancer_nuc_grouped_results.columns.values]
    cancer_nuc_grouped_results.reset_index(inplace=True)
    cancer_nuc_grouped_results['group'] = cancer_nuc_grouped_results['group'].replace({
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
        })
    
    cancer_samdcl_grouped_results = cancer_samdcl_df.groupby(['group', 'frame']).agg({
        'cell_area': ['mean', 'sem'],  # sem computes the standard error of the mean
        'cell_perimeter': ['mean', 'sem'],
        'cell_roundness': ['mean', 'sem']
    })
    cancer_samdcl_grouped_results.columns = ['_'.join(col).strip() for col in cancer_samdcl_grouped_results.columns.values]
    cancer_samdcl_grouped_results.reset_index(inplace=True)
    cancer_samdcl_grouped_results['group'] = cancer_samdcl_grouped_results['group'].replace({
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
        })
    
    plot_t_cell_segmentation_morphology_metrics(
        attached_grouped_results=attached_grouped_results,
        all_grouped_results=all_grouped_results,
        colors=colors,
        base_attached_t_cell_morphology_over_time_filename=base_attached_t_cell_morphology_over_time_filename,
        base_all_t_cell_morphology_over_time_filename=base_all_t_cell_morphology_over_time_filename,
        task_timestamp=task_timestamp
    )
    plot_individual_t_cell_segmentation_morphology_metrics(
        attached_grouped_results=attached_grouped_results,
        all_grouped_results=all_grouped_results,
        individual_colors=individual_colors,
        base_all_and_attached_t_cell_morphology_over_time_filename=base_all_and_attached_t_cell_morphology_over_time_filename,
        task_timestamp=task_timestamp
    )
    plot_individual_and_clumped_cancer_cell_segmentation_morphology_metrics(
        cancer_nuc_grouped_results=cancer_nuc_grouped_results,
        cancer_samdcl_grouped_results=cancer_samdcl_grouped_results,
        cancer_individual_colors=cancer_individual_colors,
        base_single_and_clumped_cancer_cell_morphology_over_time_filename=base_single_and_clumped_cancer_cell_morphology_over_time_filename,
        task_timestamp=task_timestamp
    )
    plot_individual_or_clumped_cancer_cell_segmentation_morphology_and_compute_summary_table_metrics(
        cancer_nuc_grouped_results=cancer_nuc_grouped_results,
        cancer_samdcl_grouped_results=cancer_samdcl_grouped_results,
        colors=colors,
        base_single_cancer_cell_morphology_over_time_filename=base_single_cancer_cell_morphology_over_time_filename,
        base_clumped_cancer_cell_morphology_over_time_filename=base_clumped_cancer_cell_morphology_over_time_filename,
        base_single_cancer_cell_morphology_over_time_table_filename=base_single_cancer_cell_morphology_over_time_table_filename,
        base_clumped_cancer_cell_morphology_over_time_table_filename=base_clumped_cancer_cell_morphology_over_time_table_filename,
        task_timestamp=task_timestamp
    )
    plot_cell_area_sum_with_ci(
        cancer_samdcl_df=cancer_samdcl_df, 
        base_clumped_cancer_cell_morphology_sum_over_time_filename=base_clumped_cancer_cell_morphology_sum_over_time_filename, 
        task_timestamp=task_timestamp, 
        colors=colors
    )
    plot_individual_and_clumped_cancer_cell_area_ratio(
        cancer_nuc_grouped_results=cancer_nuc_grouped_results,
        cancer_samdcl_grouped_results=cancer_samdcl_grouped_results,
        colors=colors,
        base_cancer_individual_clumped_area_ratio_over_time_filename=base_cancer_individual_clumped_area_ratio_over_time_filename,
        task_timestamp=task_timestamp
    )
    plot_cancer_cell_segmentation_morphology_metrics(
        cancer_samdcl_grouped_results=cancer_samdcl_grouped_results,
        colors=colors,
        base_cancer_cell_morphology_over_time_filename=base_cancer_cell_morphology_over_time_filename,
        task_timestamp=task_timestamp
    )
    make_cancer_cell_barplots(
        cancer_nuc_df,
        group_logic,
        colors,
        task_timestamp,
        base_save_single_cancer_barplots_filename,
    )
    make_cancer_cell_barplots(
        cancer_samdcl_df,
        group_logic,
        colors,
        task_timestamp,
        base_save_clumped_cancer_barplots_filename,
    )

    print(f"\nPerforming Linear Regression Tests")
    # List to hold all DataFrames for concatenation
    all_ratio_dfs = []
    # Group mapping
    groups = {
        'safe_harbor_ko': 'Safe Harbor KO',
        'rasa2_ko': 'RASA2 KO',
        'cul5_ko': 'CUL5 KO'
    }
    # Loop through each group and calculate ratios and errors
    for group, group_name in groups.items():
        # Filter data for the current group
        clumped_data = cancer_samdcl_grouped_results[cancer_samdcl_grouped_results['group'] == group_name]
        individual_data = cancer_nuc_grouped_results[cancer_nuc_grouped_results['group'] == group_name]

        # Initialize a dictionary to store frame, cell area ratios, and errors
        ratio_data = {
            'frame': clumped_data['frame'].values,
            'cell_area_ratio': [],
            'ratio_sem': []
        }
        
        # Calculate ratios and errors for each row
        for i in range(len(clumped_data)):
            ratio, error = calculate_ratio_and_error(
                clumped_data['cell_area_mean'].iloc[i], clumped_data['cell_area_sem'].iloc[i],
                individual_data['cell_area_mean'].iloc[i], individual_data['cell_area_sem'].iloc[i]
            )
            ratio_data['cell_area_ratio'].append(ratio)
            ratio_data['ratio_sem'].append(error)

        # Convert the ratio data dictionary to a DataFrame for easier manipulation/plotting
        ratio_df = pd.DataFrame(ratio_data)
        
        # Add a column to indicate the group
        ratio_df['group'] = group
        
        # Append the DataFrame to the list of all DataFrames
        all_ratio_dfs.append(ratio_df)

    # Concatenate all DataFrames into one
    combined_ratio_df = pd.concat(all_ratio_dfs, ignore_index=True)
    combined_ratio_df.drop(columns=['ratio_sem'], inplace=True)

    run_linear_regression_tests_nontracking_data(
        combined_ratio_df=combined_ratio_df, 
        cancer_nuc_df=cancer_nuc_df,
        all_t_cell_df=all_t_cell_df,
        task_timestamp=task_timestamp
    )

if __name__ == "__main__":
    test = False
    directory_path = '/gladstone/engelhardt/lab/MarsonLabIncucyteData/AnalysisFiles/CarnevaleRepStim/240106_donor2_segmentation_results/dcl_samres/'
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
    individual_colors = {
        'All T Cells': '#FF8C00',
        'Attached T Cells': '#4B0082'
    }
    cancer_individual_colors = {
        'Cancer Cell Individual': '#FF8C00',
        'Cancer Cell Clumps': '#4B0082'
    }

    cell_segmentation_morphology(
        test=test,
        directory_path=directory_path,
        group_logic=group_logic,
        colors=colors,
        individual_colors=individual_colors,
    )
