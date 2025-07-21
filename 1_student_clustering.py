# 1_student_clustering.py
#
# Task 1: Implements student sectioning using k-means clustering.
# - Reads all enrollment CSV files.
# - Creates a student-course matrix (vector space model).
# - Fits a k-means model to group students into clusters.
# - Saves two output files: a summary of cluster sizes and the detailed
#   student-to-cluster assignments.

import pandas as pd
import glob
from sklearn.cluster import KMeans
from tqdm import tqdm
import utils

def run_student_clustering(config):
    """Performs the full student clustering workflow."""
    print("--- Starting Task 1: Student Clustering ---")
    
    # 1. Load and Concatenate All Enrollment Files
    print("Loading enrollment data...")
    enrollment_files = glob.glob(config.data_paths['enrollment_files_pattern'])
    if not enrollment_files:
        print(f"Error: No enrollment files found matching pattern '{config.data_paths['enrollment_files_pattern']}'.")
        print("Please create the data files first by running '0_create_dummy_data.py'.")
        return

    df_list = [pd.read_csv(f, header=None, usecols=[1, 3, 5], encoding='latin-1') for f in enrollment_files]
    enrollments_df = pd.concat(df_list, ignore_index=True)
    enrollments_df.columns = ['StudentReg', 'CohortCode', 'CourseCode']
    print(f"Loaded {len(enrollments_df)} enrollment records from {len(enrollment_files)} files.")

    # 2. Create the Vector Space Model (Student-Course Matrix)
    print("Creating student-course matrix for clustering...")
    # Use pivot_table to create a [students x courses] matrix
    # fill_value=0 handles NaNs for students not taking a course
    student_course_matrix = enrollments_df.pivot_table(
        index='StudentReg',
        columns='CourseCode',
        aggfunc='size', # Use 'size' to count occurrences, effectively a boolean
        fill_value=0
    )
    # Ensure the matrix is binary (1 if enrolled, 0 if not)
    student_course_matrix = (student_course_matrix > 0).astype(int)
    print(f"Matrix created with shape: {student_course_matrix.shape} (students, courses)")

    # 3. Perform K-Means Clustering
    k = config.clustering_params['k_clusters']
    print(f"Fitting k-means model with k={k} clusters...")
    
    if student_course_matrix.shape[0] < k:
        print(f"Warning: Number of students ({student_course_matrix.shape[0]}) is less than k ({k}). Adjusting k.")
        k = student_course_matrix.shape[0]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    
    # Use tqdm to show progress for large datasets
    with tqdm(total=1, desc="Clustering") as pbar:
        cluster_labels = kmeans.fit_predict(student_course_matrix)
        pbar.update(1)
    
    print("Clustering complete.")

    # 4. Generate Output Files
    # Create a mapping from StudentReg to their assigned ClusterID
    student_to_cluster = pd.DataFrame({
        'StudentReg': student_course_matrix.index,
        'ClusterID': cluster_labels
    })

    # Output 1: Cluster Summary
    cluster_summary = student_to_cluster['ClusterID'].value_counts().reset_index()
    cluster_summary.columns = ['ClusterID', 'ClusterSize']
    cluster_summary.to_csv(config.output_paths['cluster_summary'], index=False)
    print(f"Cluster summary saved to {config.output_paths['cluster_summary']}")

    # Output 2: Detailed Student Assignments
    # Merge the cluster IDs back into the original, full enrollment data
    student_assignments = enrollments_df.merge(student_to_cluster, on='StudentReg')
    student_assignments.to_csv(config.output_paths['student_cluster_assignments'], index=False)
    print(f"Detailed student cluster assignments saved to {config.output_paths['student_cluster_assignments']}")
    print("--- Task 1 Complete ---")

if __name__ == "__main__":
    # --- 1. Get the Configuration for the Active Session ---
    config = utils.get_config()
    print(f"--- Running student clustering for session: {config.session_name} ---")
    utils.setup_directories(config)
    run_student_clustering(config)