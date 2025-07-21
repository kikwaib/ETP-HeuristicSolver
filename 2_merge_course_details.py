# 2_merge_course_details.py
#
# Task 2: Merges course information with department and school information.
# - Reads '01-all-units-courses.csv' and 'departments-mod-2.csv'.
# - Performs a case-insensitive merge on the department name.
# - Creates a clean, unified 'course_details.csv' file.

import pandas as pd
import utils

def run_course_department_merge(config):
    """Performs the full course and department merge workflow."""
    print("--- Starting Task 2: Merge Course and Department Details ---")

    # 1. Load Input Files
    try:
        courses_df = pd.read_csv(config.data_paths['all_units_courses'], usecols=[0, 1, 2], header=None, encoding='latin-1')
        courses_df.columns = ['CourseCode', 'CourseName', 'DeptName_course_file']

        depts_df = pd.read_csv(config.data_paths['departments'], encoding='latin-1')
    except FileNotFoundError as e:
        print(f"Error: Could not find input file {e.filename}.")
        print("Please create the data files first by running '0_create_dummy_data.py'.")
        return

    print("Loaded course and department files.")

    # 2. Prepare for Case-Insensitive Merge
    # Create temporary lowercase columns to join on
    courses_df['join_key'] = courses_df['DeptName_course_file'].str.lower()
    depts_df['join_key'] = depts_df['Dept Name'].str.lower()

    # 3. Perform the Merge
    # Use a left merge to keep all courses, even if a department is not found
    merged_df = pd.merge(
        courses_df,
        depts_df,
        on='join_key',
        how='left'
    )
    print("Merge complete.")

    # 4. Clean Up and Format the Final DataFrame
    # Select the required columns, using the proper-cased name from the depts file
    final_df = merged_df[[
        'CourseCode',
        'CourseName',
        'Dept Name',  # The canonical department name
        'Dept Code',
        'School Name',
        'School Code',
        'MPreferred' 
    ]]
    
    # Rename columns for clarity and consistency
    final_df = final_df.rename(columns={
        'Dept Name': 'DeptName',
        'Dept Code': 'DeptCode',
        'School Name': 'SchName',
        'School Code': 'SchCode'
    })

    # Handle cases where a department in the course file was not in the departments file
    unmatched_count = final_df['DeptName'].isnull().sum()
    if unmatched_count > 0:
        print(f"Warning: {unmatched_count} courses had department names that could not be matched.")
        # For robustness, fill missing DeptName with the original name from the course file
        final_df['DeptName'] = final_df['DeptName'].fillna(merged_df['DeptName_course_file'])

    # 5. Save the Output
    final_df['CrosslistedID'] = ''
    final_df.to_csv(config.output_paths['course_details'], index=False)
    print(f"Unified course details saved to {config.output_paths['course_details']}")
    print("--- Task 2 Complete ---")

if __name__ == "__main__":
    # --- Get the Configuration for the Active Session ---
    config = utils.get_config()
    print(f"--- Merging Course and Department Details for Session: {config.session_name} ---")
    run_course_department_merge(config)