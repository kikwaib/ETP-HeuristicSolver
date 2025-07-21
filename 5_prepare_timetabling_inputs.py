# 5_prepare_timetabling_inputs.py
#
# Task 5: Generates the final set of CSV files required as input for the
# heuristic timetabling solver. It pre-calculates conflicts and aggregates
# data into clean, ready-to-use formats.

import pandas as pd
import glob
import sys
from itertools import combinations
from collections import Counter
from tqdm import tqdm
import utils

def generate_conflict_graph(enrollments_df):
    """
    Generates a graph of conflicting exams, weighted by the number of
    students creating the conflict.
    """
    print("\n--- Generating Exam Conflict Graph ---")
    
    # Group enrollments by student
    student_courses = enrollments_df.groupby('StudentReg')['CourseCode'].apply(list)
    
    # Use a Counter for highly efficient counting of pairs
    conflict_counter = Counter()

    print("Calculating all conflicting pairs...")
    for courses in tqdm(student_courses, desc="Processing Students"):
        # A student only creates conflicts if they take 2 or more exams
        if len(courses) >= 2:
            # Sort the courses to ensure ('A', 'B') is the same as ('B', 'A')
            sorted_courses = sorted(courses)
            # Generate all unique pairs of exams for this student
            for pair in combinations(sorted_courses, 2):
                conflict_counter[pair] += 1
    
    if not conflict_counter:
        print("Warning: No conflicting exam pairs found.")
        return pd.DataFrame(columns=['ExamCode1', 'ExamCode2', 'ConflictStrength'])
        
    # Convert the counter to a DataFrame
    conflict_list = [
        {'ExamCode1': pair[0], 'ExamCode2': pair[1], 'ConflictStrength': count}
        for pair, count in conflict_counter.items()
    ]
    
    conflict_df = pd.DataFrame(conflict_list)
    print(f"Generated conflict graph with {len(conflict_df)} unique edges (conflicting pairs).")
    return conflict_df

def generate_cluster_enrollments(student_assignments_df):
    """
    Aggregates student enrollments at the cluster level.
    """
    print("\n--- Generating Cluster-Exam Enrollments ---")
    
    if student_assignments_df.empty:
        print("Warning: Student assignments DataFrame is empty. Cannot generate cluster enrollments.")
        return pd.DataFrame(columns=['ClusterID', 'ExamCode', 'NumStudents'])

    # Group by Cluster and Course, then count the number of students
    cluster_enrollments = student_assignments_df.groupby(['ClusterID', 'CourseCode']) \
                                                .size() \
                                                .reset_index(name='NumStudents')
    
    print(f"Generated {len(cluster_enrollments)} cluster-exam enrollment records.")
    return cluster_enrollments


def generate_exam_master_list(course_details_df, enrollments_df):
    """
    Creates a single, definitive master file for all exams.
    UPDATED to add the School column, drop zero-enrollment exams,
    and ENSURE EXAMCODE UNIQUENESS.
    """
    print("\n--- Generating Exam Master List ---")

    # --- NEW: Drop duplicates from the source course details first ---
    # This prevents the primary key violation error in the solver.
    # We keep the 'first' entry we see for any given CourseCode.
    original_len = len(course_details_df)
    course_details_df = course_details_df.drop_duplicates(subset=['CourseCode'], keep='first')
    if len(course_details_df) < original_len:
        print(f"Warning: Removed {original_len - len(course_details_df)} duplicate course codes from course_details.csv.")

    # Calculate total enrollment for each exam using unique students
    total_enrollments = enrollments_df.groupby('CourseCode')['StudentReg'] \
                                      .nunique() \
                                      .reset_index(name='TotalEnrollment')
    
    # Merge total enrollments into the course details dataframe
    exam_master_df = pd.merge(
        course_details_df,
        total_enrollments,
        on='CourseCode',
        how='left'
    )
    
    # Fill any courses with no students with an enrollment of 0
    exam_master_df['TotalEnrollment'] = exam_master_df['TotalEnrollment'].fillna(0).astype(int)
    
    # Filter out exams with zero enrollment
    original_count = len(exam_master_df)
    exam_master_df = exam_master_df[exam_master_df['TotalEnrollment'] > 0].copy()
    print(f"Dropped {original_count - len(exam_master_df)} exams with zero enrollment.")
    
    # Ensure the column order is correct, now including School Name
    final_columns = [
        'ExamCode', 'CourseName', 'DeptName', 'SchName', 'TotalEnrollment',
        'MPreferred', 'CrosslistedID'
    ]
    
    # Rename CourseCode to ExamCode for consistency
    exam_master_df = exam_master_df.rename(columns={'CourseCode': 'ExamCode'})

    # Reorder the columns to match the desired output
    exam_master_df = exam_master_df[final_columns]
    
    print(f"Generated exam master list with {len(exam_master_df)} unique exams.")
    return exam_master_df


def prepare_timetabling_inputs(config):
    """Main orchestrator function to load data and generate all files."""
    print("--- Starting Final Preparation of Timetabling Inputs ---")

    # 1. Load all required source files
    try:
        # Load all raw enrollment files for conflict graph and total enrollments
        enrollment_files = glob.glob(config.data_paths['enrollment_files_pattern'])
        if not enrollment_files:
            raise FileNotFoundError("No raw enrollment files found.")
        df_list = [pd.read_csv(f, header=None, encoding='latin1', usecols=[1, 5]) for f in enrollment_files]
        raw_enrollments_df = pd.concat(df_list, ignore_index=True)
        raw_enrollments_df.columns = ['StudentReg', 'CourseCode']

        student_assignments_df = pd.read_csv(config.output_paths['student_cluster_assignments'], encoding='latin1')
        course_details_df = pd.read_csv(config.output_paths['course_details'], encoding='latin1')
    except FileNotFoundError as e:
        print(f"Error: A required input file is missing: {e.filename}", file=sys.stderr)
        print("Please ensure scripts 1 and 2 have been run successfully.", file=sys.stderr)
        sys.exit(1)
        
    print("All source files loaded successfully.")

    # 2. Generate each of the three required files
    conflict_df = generate_conflict_graph(raw_enrollments_df)
    cluster_enrollments_df = generate_cluster_enrollments(student_assignments_df)
    exam_master_df = generate_exam_master_list(course_details_df, raw_enrollments_df)

    # 3. Save all generated files
    conflict_df.to_csv(config.output_paths['exam_conflict_graph'], index=False)
    print(f"Conflict graph saved to {config.output_paths['exam_conflict_graph']}")

    cluster_enrollments_df.to_csv(config.output_paths['cluster_exam_enrollments'], index=False)
    print(f"Cluster enrollments saved to {config.output_paths['cluster_exam_enrollments']}")

    exam_master_df.to_csv(config.output_paths['exam_master_list'], index=False)
    print(f"Exam master list saved to {config.output_paths['exam_master_list']}")

    print("\n--- All timetabling input files have been successfully generated. ---")


if __name__ == "__main__":
    # --- Get the Configuration for the Active Session ---
    config = utils.get_config()
    print(f"--- Preparing Timetabling Inputs for Session: {config.session_name} ---")
    utils.setup_directories(config)
    prepare_timetabling_inputs(config)