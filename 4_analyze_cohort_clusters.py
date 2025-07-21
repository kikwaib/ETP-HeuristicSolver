# 4_analyze_cohort_clusters.py
#
# Task 4: Analyzes the student cluster assignments to provide a cohort-level summary.
# - Reads the detailed 'student_cluster_assignments.csv' file.
# - Groups the data by 'CohortCode'.
# - For each cohort, it calculates the number of unique clusters its students
#   belong to and lists those cluster IDs.
# - Saves the aggregated results to 'cohort_cluster_analysis.csv'.

import pandas as pd
import sys
import utils

def run_cohort_analysis(config):
    """Performs the full cohort-level cluster analysis."""
    print("--- Starting Task 4: Analyze Cohort-Cluster Distribution ---")

    # 1. Load the required input file from Task 1
    input_path = config.output_paths['student_cluster_assignments']
    try:
        assignments_df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'.")
        print("Please run '1_student_clustering.py' first to generate the necessary file.")
        sys.exit(1)
        
    print(f"Loaded data from {input_path}.")

    # 2. Group by 'CohortCode' and perform aggregations
    print("Aggregating cluster information by cohort...")

    # Define the aggregation functions for the 'ClusterID' column
    # - 'nunique' to count the unique clusters
    # - A custom lambda function to create the comma-separated string
    agg_functions = {
        'ClusterID': [
            ('CL_Number', 'nunique'),
            ('ClusterIDs', lambda ids: ','.join(sorted(ids.unique().astype(str))))
        ]
    }

    # Perform the groupby and aggregation
    cohort_analysis = assignments_df.groupby('CohortCode').agg(agg_functions)

    # The result has a multi-level column index, so we need to flatten it
    cohort_analysis.columns = cohort_analysis.columns.droplevel(0)
    
    # Reset the index to turn 'CohortCode' from an index back into a column
    cohort_analysis = cohort_analysis.reset_index()

    print("Aggregation complete.")

    # 3. Save the final output file
    output_path = config.output_paths['cohort_cluster_analysis']
    cohort_analysis.to_csv(output_path, index=False)
    
    print(f"\nCohort cluster analysis results:")
    print(cohort_analysis.to_string())
    print(f"\nAnalysis saved to {output_path}")
    print("--- Task 4 Complete ---")

if __name__ == "__main__":
    # --- Get the Configuration for the Active Session ---
    config = utils.get_config()
    print(f"--- Analyzing Cohort Clusters for Session: {config.session_name} ---")
    # Ensure directories exist before running
    utils.setup_directories(config)
    run_cohort_analysis(config)