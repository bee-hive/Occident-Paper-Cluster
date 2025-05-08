"""
This file contains helper functions that are specific to the SH, RASA2, and CUL5 dataset. It is meant to be complementary
to the functions in occident/tracking.py, occident/velocity.py, and occident/utils.py.
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind_from_stats
from scipy.stats import linregress
import statsmodels.formula.api as smf
from occident.tracking import get_group_from_filename
from occident.utils import mean_confidence_interval, estimate_se

def make_roundness_plot_and_table(
        input_df,
        group_logic,
        colors,
        task_timestamp,
        base_save_roundness_table_filename,
        base_save_roundness_plot_filename
):
    """
    Generates a roundness plot and summary table for T cell data.

    This function calculates the mean roundness and confidence intervals for different groups of T cells
    based on the input DataFrame. It performs pairwise t-tests between the groups to determine statistical 
    significance and calculates z-scores. The results are saved in a CSV file and a bar plot of the average 
    roundness with 95% confidence interval is generated and saved.

    Parameters:
    - input_df: DataFrame containing T cell data, including 't_cell_id', 't_cell_roundness', and 'filename'.
    - group_logic: Logic for grouping the data based on the filename.
    - colors: Dictionary mapping group names to specific colors for plotting.
    - task_timestamp: Timestamp used for saving the output files.
    - base_save_roundness_table_filename: Base filename for saving the summary table CSV.
    - base_save_roundness_plot_filename: Base filename for saving the roundness plot.

    Outputs:
    - Saves a CSV file with the summary statistics and statistical test results.
    - Saves a bar plot image file showing the average roundness with confidence intervals by group.
    """

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
    """
    Generates a bar plot of the mean and 95% CI of the cancer cell metrics (cell_area, cell_perimeter, cell_roundness) 
    for each group of cancer cells.

    Parameters:
    - input_df: DataFrame containing cancer cell data, including 'cell_id', 'frame', 'filename', and the metric of interest.
    - group_logic: Logic for grouping the data based on the filename.
    - colors: Dictionary mapping group names to specific colors for plotting.
    - task_timestamp: Timestamp used for saving the output files.
    - base_save_cancer_barplots_filename: Base filename for saving the bar plots.

    Outputs:
    - Saves a bar plot image file showing the average of the given metric with 95% CI by group for each metric.
    """
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

def compare_groups(
        results_df, 
        base_group_name, 
        subset_comparison_groups
):
    """
    Compare statistical metrics between a base group and multiple comparison groups.

    This function performs pairwise t-tests between a specified base group and a subset of comparison groups
    for each metric and group present in the results DataFrame. It calculates the t-statistic, p-value, and
    applies a Bonferroni correction to adjust for multiple comparisons.

    Parameters:
    - results_df: pd.DataFrame
        DataFrame containing the results data with columns for 'metric', 'group', 'mean', '95_ci_lower',
        '95_ci_upper', and 'n' for each frame group.
    - base_group_name: str
        The name of the frame group to be used as the base for comparisons.
    - subset_comparison_groups: list of str
        A list of frame group names to be compared against the base group.

    Returns:
    - pd.DataFrame
        A DataFrame containing the comparison results with columns for 'base_group', 'comparison_group',
        'metric', 'group', 'base_mean', 'base_lower_ci', 'base_upper_ci', 'base_se', 'base_n',
        'comparison_mean', 'comparison_lower_ci', 'comparison_upper_ci', 'comparison_se', 'comparison_n',
        'p_value', 'z_score', and 'bonferroni_correction'.
    """

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
    """
    Function to generate comparison tables for cell tracking data.

    Parameters
    ----------
    input_df : pandas.DataFrame
        DataFrame containing cell tracking data.
    task_timestamp : str
        Timestamp from when the data was generated. Used to generate filename.
    base_filename : str
        Base filename for the output files.

    Returns
    -------
    None

    Notes
    -----
    This function generates comparison tables between different frame groups. The comparisons are calculated as follows:
    - For each frame group, calculate the mean and 95% confidence intervals for each metric.
    - Perform a two-sample t-test to compare the means of each metric between the frame groups.
    - Apply a Bonferroni correction to the p-values.
    """
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
    """
    Calculates linear regression metrics for specified columns in a DataFrame.

    For each column in `y_cols`, this function performs a linear regression with `x_col`
    as the independent variable and calculates the slope, intercept, standard error, and
    95% confidence intervals for the slope.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data for regression analysis.
    x_col : str
        Column name in `df` to be used as the independent variable.
    y_cols : list of str
        List of column names in `df` to be used as dependent variables.

    Returns
    -------
    dict
        A dictionary where each key is a column name from `y_cols`, and each value is
        another dictionary containing the regression metrics: 'slope', 'intercept', 'se',
        '95_ci_lower', and '95_ci_upper'.
    """

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
    """
    Function to generate comparison tables for linear regression analysis of cell tracking data.

    Parameters
    ----------
    input_df : pandas.DataFrame
        DataFrame containing cell tracking data.
    task_timestamp : str
        Timestamp from when the data was generated. Used to generate filename.
    base_linear_regression_filename : str
        Base filename for the output file.

    Returns
    -------
    None

    Notes
    -----
    This function generates comparison tables for linear regression analysis of cell tracking data. The comparisons are calculated as follows:
    - For each frame group, calculate the mean and 95% confidence intervals for each metric.
    - Perform a two-sample t-test to compare the means of each metric between the frame groups.
    - Apply a Bonferroni correction to the p-values.
    """
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

def make_interaction_analysis_plots(
        df_consecutive_interactions_dict,
        colors,
        task_timestamp,
        base_unique_interaction_table_filename,
        base_interaction_distribution_table_filename,
        base_interaction_distribution_plot_filename,
        base_interaction_distribution_over_time_plot_filename
):
    """
    Function to generate various analysis plots and tables from a DataFrame
    containing consecutive frames of cell interactions.

    Parameters
    ----------
    df_consecutive_interactions_dict : dict
        Dictionary containing DataFrames with consecutive frames of cell interactions.
    colors : dict
        Dictionary containing colors to use for each group in the plots.
    task_timestamp : str
        Timestamp to include in the filenames of the output files.
    base_unique_interaction_table_filename : str
        Base filename for the unique interactions table.
    base_interaction_distribution_table_filename : str
        Base filename for the interaction distribution table.
    base_interaction_distribution_plot_filename : str
        Base filename for the interaction distribution plot.
    base_interaction_distribution_over_time_plot_filename : str
        Base filename for the interaction distribution over time plot.

    Returns
    -------
    None
    """
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
    """
    Plot T Cell or Cancer Cell metrics subplots for ≥2, ≥5, and ≥10 minimum consecutive frames.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot, with columns 'group', 'consec_frame_group', 'interaction_id_consec_frame', and the desired metrics (e.g. 't_cell_perimeter', 't_cell_area', 't_cell_roundness', 't_cell_velocity').
    colors : dict
        Dictionary mapping group names to colors for the plot.
    plot_filename : str
        Filename to save the plot to.

    Notes
    -----
    The function will create a figure with 4x3 subplots to accommodate the additional metric row, and share the x-axis between the columns. The y-axis limits are set to the minimum and maximum of the data for each row, and the legend is only shown in the first plot of each row. The plot is saved to the specified filename.

    """
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
        plot_filename_base,
        timestamp
):
    """
    Plots individual metrics (Perimeter, Area, Roundness, and Velocity) for T Cells or Cancer Cells over consecutive frames.

    This function generates line plots for specified metrics across different groups of cell interactions, separated by 
    minimum consecutive frames. The plots are saved as PDF files.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot, with columns 'group', 'consec_frame_group', 'interaction_id_consec_frame', 
        and the desired metrics (e.g., 't_cell_perimeter', 't_cell_area', 't_cell_roundness', 't_cell_velocity').
    colors : dict
        A dictionary mapping group names to specific colors used in plotting.
    plot_filename_base : str
        Base filename to append details to for saving each plot.
    timestamp : str
        A timestamp string to include in the filename for each plot for uniqueness.

    Notes
    -----
    - The function standardizes group names for consistency in plotting.
    - Plots include error bands representing a 95% confidence interval.
    - Each metric for each group of consecutive frames is saved as an individual plot.
    """
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
    """
    Plot mean morphology metrics for T cells over time, comparing attached and all T cells.

    Parameters
    ----------
    attached_grouped_results : DataFrame
        Dataframe with grouped results for attached T cells.
    all_grouped_results : DataFrame
        Dataframe with grouped results for all T cells.
    colors : dict
        Mapping of group names to colors.
    base_attached_t_cell_morphology_over_time_filename : str
        Base filename for saving the plots.
    base_all_t_cell_morphology_over_time_filename : str
        Base filename for saving the plots.
    task_timestamp : str
        Timestamp for the task.

    Returns
    -------
    None
    """
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
    """
    Plots the mean of a specified metric over frames for different groups within a DataFrame.
    Helper function for plot_t_cell_segmentation_morphology_metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot, with columns 'group', 'frame', and '{metric}_mean' and '{metric}_sem'.
    metric : str
        The metric to plot (e.g., 'cell_area', 'cell_perimeter', 'cell_roundness').
    title : str
        The title for the y-axis of the plot.
    colors : dict
        A dictionary mapping group names to specific colors used in plotting.
    ax : matplotlib.axes.Axes
        The matplotlib axes object where the plot will be drawn.

    Notes
    -----
    - The function creates a line plot for each group in the DataFrame, with shaded error bands representing
      the standard error of the mean (SEM).
    - The x-axis represents the 'frame', and the y-axis represents the mean value of the specified metric.
    """

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
    """
    Plots individual metrics (Perimeter, Area, Roundness) for T cells (attached and all) over consecutive frames.

    This function generates line plots for specified metrics across different groups of cell interactions, separated by 
    minimum consecutive frames. The plots are saved as PDF files.

    Parameters
    ----------
    attached_grouped_results : pandas.DataFrame
        DataFrame containing grouped results for attached T cells.
    all_grouped_results : pandas.DataFrame
        DataFrame containing grouped results for all T cells.
    individual_colors : dict
        Mapping of group names to specific colors used in plotting.
    base_all_and_attached_t_cell_morphology_over_time_filename : str
        Base filename for saving the plots.
    task_timestamp : str
        Timestamp for the task.

    Notes
    -----
    - The function standardizes group names for consistency in plotting.
    - Plots include error bands representing a 95% confidence interval.
    - Each metric for each group of consecutive frames is saved as an individual plot.
    """
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
    """
    Plots individual metrics (Perimeter, Area, Roundness) for Cancer Cells (clumped and individual) over consecutive frames.

    This function generates line plots for specified metrics across different groups of cell interactions, separated by 
    minimum consecutive frames. The plots are saved as PDF files.

    Parameters
    ----------
    cancer_nuc_grouped_results : pandas.DataFrame
        DataFrame containing grouped results for Cancer Cells.
    cancer_samdcl_grouped_results : pandas.DataFrame
        DataFrame containing grouped results for Cancer Cell Clumps.
    cancer_individual_colors : dict
        Mapping of group names to specific colors used in plotting.
    base_single_and_clumped_cancer_cell_morphology_over_time_filename : str
        Base filename for saving the plots.
    task_timestamp : str
        Timestamp for the task.

    Notes
    -----
    - The function standardizes group names for consistency in plotting.
    - Plots include error bands representing a 95% confidence interval.
    - Each metric for each group of consecutive frames is saved as an individual plot.
    """
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
    """
    Plots cancer cell segmentation morphology metrics and computes summary statistics for individual and clumped cells.

    This function creates line plots for specified metrics (Area, Perimeter, Roundness) across different groups of 
    cancer cell interactions, both for individual and clumped cells. It computes and saves summary statistics 
    (mean, upper and lower 95% confidence intervals, and standard error) to CSV files, and saves the plots as PDF files.

    Parameters
    ----------
    cancer_nuc_grouped_results : pandas.DataFrame
        DataFrame containing grouped results for individual cancer cells.
    cancer_samdcl_grouped_results : pandas.DataFrame
        DataFrame containing grouped results for clumped cancer cells.
    colors : dict
        Mapping of group names to specific colors used in plotting.
    base_single_cancer_cell_morphology_over_time_filename : str
        Base filename for saving plots of individual cancer cells.
    base_clumped_cancer_cell_morphology_over_time_filename : str
        Base filename for saving plots of clumped cancer cells.
    base_single_cancer_cell_morphology_over_time_table_filename : str
        Base filename for saving summary statistics of individual cancer cells.
    base_clumped_cancer_cell_morphology_over_time_table_filename : str
        Base filename for saving summary statistics of clumped cancer cells.
    task_timestamp : str
        Timestamp for the task to ensure unique filenames.

    Notes
    -----
    - The function standardizes group names for consistency in plotting.
    - Plots include error bands representing a 95% confidence interval.
    - Summary statistics are calculated and saved for each metric and group type.
    """
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
    """
    Plot the mean of a specified metric over frames for a single group within a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot, with columns 'frame', '{metric}_mean', and '{metric}_sem'.
    metric : str
        The metric to plot (e.g., 'cell_area', 'cell_perimeter', 'cell_roundness').
    title : str
        The title for the y-axis of the plot.
    color : str
        The color for the line and error shading.
    label : str
        The label for the legend.
    ax : matplotlib.axes.Axes
        The matplotlib axes object where the plot will be drawn.

    Notes
    -----
    - The function creates a line plot for the group, with shaded error bands representing
      the standard error of the mean (SEM).
    - The x-axis represents the 'frame', and the y-axis represents the mean value of the specified metric.
    """
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
    """
    Plots the sum of cell areas over time with 95% confidence intervals for different groups of clumped cancer cells.

    This function processes a DataFrame containing cancer cell data, calculates the sum of cell areas 
    for each well, and then computes the mean and standard error of the mean (SEM) for each group of cells 
    over time. It plots these values with shaded error bands representing the 95% confidence intervals.

    Parameters
    ----------
    cancer_samdcl_df : pandas.DataFrame
        DataFrame containing data of clumped cancer cells with columns 'frame', 'group', 'filename', and 'cell_area'.
    base_clumped_cancer_cell_morphology_sum_over_time_filename : str
        Base filename for saving the plot.
    task_timestamp : str
        Timestamp for the task, used to ensure unique filenames.
    colors : dict
        Mapping of group names to specific colors used in plotting.

    Notes
    -----
    - The function standardizes group names for consistency in plotting.
    - The plot includes error bands representing a 95% confidence interval.
    - The plot is saved as a PDF file.
    """

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
    """
    Plots the ratio of cell areas between clumped and individual cancer cells over time for different groups.

    This function processes two DataFrames containing grouped results for individual and clumped cancer cells,
    calculates the ratio of cell areas and errors for each group over time, and then plots these values with
    shaded error bands representing the standard error of the mean (SEM).

    Parameters
    ----------
    cancer_nuc_grouped_results : pandas.DataFrame
        DataFrame containing grouped results for individual cancer cells.
    cancer_samdcl_grouped_results : pandas.DataFrame
        DataFrame containing grouped results for clumped cancer cells.
    colors : dict
        Mapping of group names to specific colors used in plotting.
    base_cancer_individual_clumped_area_ratio_over_time_filename : str
        Base filename for saving the plot.
    task_timestamp : str
        Timestamp for the task, used to ensure unique filenames.

    Notes
    -----
    - The function standardizes group names for consistency in plotting.
    - The plot includes error bands representing a 95% confidence interval.
    - The plot is saved as a PDF file.
    """
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
    """
    Plot mean morphology metrics for cancer cells over time, comparing different groups.

    Parameters
    ----------
    cancer_samdcl_grouped_results : DataFrame
        Dataframe with grouped results for cancer cells.
    colors : dict
        Mapping of group names to colors.
    base_cancer_cell_morphology_over_time_filename : str
        Base filename for saving the plots.
    task_timestamp : str
        Timestamp for the task.

    Returns
    -------
    None
    """
    # Standardize group names
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
    
    """
    Runs linear regression tests on a set of T cell metrics.

    Parameters:
    ----------
    cell_combined_dict : dict
        A dictionary containing the combined data for each group.
    task_timestamp : str
        A string representing the timestamp of the task.
    base_linear_regression_model_summary_table : str
        A string representing the base filename for the linear regression model summary table.

    Returns:
    None
    """
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
    save_path = f"./plots/clumped_to_single_cancer_cell_area_ratio_linear_regression_model_summary_{task_timestamp}.csv"
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
    save_path = f"./tables/all_t_cell_roundness_linear_regression_model_summary_{task_timestamp}.csv"
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
    save_path = f"./tables/all_t_cell_roundness_linear_regression_model_summary_{task_timestamp}.csv"
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
    save_path = f"./tables/single_cancer_cell_roundness_linear_regression_model_summary_{task_timestamp}.csv"
    save_path = os.path.expanduser(save_path)
    model_summary_df.to_csv(save_path, index=False)