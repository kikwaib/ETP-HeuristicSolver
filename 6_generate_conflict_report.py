# 6_generate_conflict_report.py
#
# Task 6: A comprehensive post-processing and validation script for the
# heuristic solver's output.
# - Reads the completed schedule from final_schedule.csv.
# - Cross-references it with all original data sources.
# - Generates a detailed report of all students who have a direct conflict.
# - Generates a summary validation_report.txt that checks all hard constraints.

import pandas as pd
import glob
import sys
from tqdm import tqdm
import collections
import utils
import os
from datetime import datetime

def run_final_validation(config):
    """
    Analyzes a final schedule for a specific run to find conflicts and
    validate all hard constraints.
    """
    print(f"\n--- Validating Schedule for Run ID: {config.run_timestamp} ---")
    
    # --- 1. Load the specific, timestamped final schedule ---
    try:
        schedule_path = config.output_paths['final_schedule']
        print(f"Loading schedule: {schedule_path}")
        schedule_df = pd.read_csv(schedule_path)
        
        # Load other required files (these are not timestamped)
        student_assignments_df = pd.read_csv(config.output_paths['student_cluster_assignments'])
        exam_master_df = pd.read_csv(config.output_paths['exam_master_list'])
        rooms_df = pd.read_csv(config.data_paths['rooms'])

        enrollment_files = glob.glob(config.data_paths['enrollment_files_pattern'])
        df_list = [pd.read_csv(f, header=None, encoding="latin1", usecols=[1, 5]) for f in enrollment_files]
        raw_enrollments_df = pd.concat(df_list, ignore_index=True)
        raw_enrollments_df.columns = ['StudentReg', 'CourseCode']
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find schedule file: {schedule_path}", file=sys.stderr)
        print("Please ensure the timestamp is correct and the solver has been run.", file=sys.stderr)
        sys.exit(1)

    print("All necessary files for validation have been loaded.")
    report_lines = [f"--- Validation Report for Run: {config.run_timestamp} ---"]

    # --- Check for and report unplaced exams first ---
    unplaced_report_path = config.output_paths['unplaced_exams_report']
    try:
        if os.path.exists(unplaced_report_path) and os.path.getsize(unplaced_report_path) > 0:
            unplaced_df = pd.read_csv(unplaced_report_path)
            report_lines.append(f"\nWARNING: {len(unplaced_df)} EXAMS COULD NOT BE PLACED.")
            report_lines.append("This indicates the problem may be infeasible under current constraints.")
            report_lines.append("Unplaced Exams (ExamCode, EnrollmentSize):")
            for _, row in unplaced_df.iterrows():
                report_lines.append(f"  - {row['ExamCode']} (Size: {row['TotalEnrollment']})")
        else:
            report_lines.append("\nPASS: All exams from the master list were successfully placed in the schedule.")
    except FileNotFoundError:
        report_lines.append("\nINFO: No unplaced exams report was found. Assuming all exams were placed.")

    # --- Validation 1: Generate the Student Conflict Report ---
    print("\nAnalyzing schedule to find student conflicts...")
    
    # Create a lookup for which exams are in which slot
    slot_to_exams = schedule_df.groupby(['Day', 'Period'])['ExamCode'].apply(set).to_dict()
    
    # Create a lookup for each student's full schedule
    student_schedules = raw_enrollments_df.groupby('StudentReg')['CourseCode'].apply(set).to_dict()

    conflict_records = []
    for student_id, courses in tqdm(student_schedules.items(), desc="Checking Student Conflicts"):
        # For each slot, check how many of the student's courses are scheduled
        for slot, exams_in_slot in slot_to_exams.items():
            conflicting_courses = courses.intersection(exams_in_slot)
            
            if len(conflicting_courses) > 1:
                conflict_records.append({
                    'StudentReg': student_id,
                    'TimeSlot': f"{slot[0]} {slot[1]}",
                    'NumConflicts': len(conflicting_courses),
                    'ConflictingExams': ','.join(sorted(list(conflicting_courses)))
                })

    if not conflict_records:
        report_lines.append("\nPASS: No student conflicts found in the final schedule.")
        # Create an empty file to indicate completion
        pd.DataFrame(columns=['StudentReg', 'ClusterID', 'TimeSlot', 'NumConflicts', 'ConflictingExams']) \
          .to_csv(config.output_paths['student_conflict_report'], index=False)
    else:
        conflict_report_df = pd.DataFrame(conflict_records)
        student_to_cluster = student_assignments_df[['StudentReg', 'ClusterID']].drop_duplicates()
        conflict_report_df = pd.merge(conflict_report_df, student_to_cluster, on='StudentReg', how='left')

        output_path = config.output_paths['student_conflict_report']
        conflict_report_df.to_csv(output_path, index=False)
        
        report_lines.append(f"\nFAIL [SOFT]: Found {len(conflict_report_df)} instances of student conflicts.")
        report_lines.append(f"             A detailed list has been saved to: {output_path}")

    # --- Validation 2: Check Hard Constraints ---
    print("\nValidating hard constraints...")
    
    # 2a: Check Concurrency for Cross-Listed Exams
    crosslist_groups = exam_master_df[exam_master_df['CrosslistedID'].notna() & (exam_master_df['CrosslistedID'] != '')].groupby('CrosslistedID')
    concurrency_fail = False
    schedule_lookup = schedule_df.set_index('ExamCode')
    for cl_id, group in crosslist_groups:
        slots = set()
        for exam_id in group['ExamCode']:
            if exam_id in schedule_lookup.index:
                row = schedule_lookup.loc[exam_id]
                slots.add((row['Day'], row['Period']))
        if len(slots) > 1:
            report_lines.append(f"FAIL [HARD]: Cross-listed group '{cl_id}' is not concurrent. Found in slots: {slots}")
            concurrency_fail = True
    if not concurrency_fail:
        report_lines.append("PASS: All cross-listed exams are scheduled concurrently.")

    # 2b: Check Room Capacity
    capacity_fail = False
    room_capacities = rooms_df.set_index('RoomID')['Capacity'].to_dict()
    # Group by both slot and room to get total enrollment in each specific room
    room_loads = schedule_df.merge(exam_master_df[['ExamCode', 'TotalEnrollment']], on='ExamCode') \
                            .groupby(['Day', 'Period', 'RoomID'])['TotalEnrollment'].sum().reset_index()

    for _, row in room_loads.iterrows():
        room_id = row['RoomID']
        total_load = row['TotalEnrollment']
        capacity = room_capacities.get(room_id, 0)
        if total_load > capacity:
            report_lines.append(f"FAIL [HARD]: Room {room_id} in slot ({row['Day']}, {row['Period']}) is over capacity. Load: {total_load}, Capacity: {capacity}")
            capacity_fail = True
    if not capacity_fail:
        report_lines.append("PASS: All room capacities are respected.")
        
    # 2c: Check if all exams were scheduled
    scheduled_exams = set(schedule_df['ExamCode'].unique())
    all_exams = set(exam_master_df['ExamCode'].unique())
    unscheduled = all_exams - scheduled_exams
    if unscheduled:
        report_lines.append(f"FAIL [HARD]: {len(unscheduled)} exams were not scheduled: {unscheduled}")
    else:
        report_lines.append("PASS: All exams from the master list have been scheduled.")

    # --- Final Report Generation ---
    final_report_str = "\n".join(report_lines)
    print("\n" + final_report_str)
    
    # Save reports using their timestamped paths from the config
    with open(config.output_paths['validation_report'], 'w') as f:
        f.write(final_report_str)
    print(f"\nValidation report saved to {config.output_paths['validation_report']}")
    print("--- Final Validation Complete ---")

if __name__ == "__main__":
    # --- Get the base config to know which session we are in ---
    base_config = utils.get_config()
    print(f"--- Running Validation for Session: {base_config.session_name} ---")
    
    # --- Prompt user for the timestamp ---
    user_timestamp = input(f"Please enter the run timestamp to validate (e.g., {datetime.now().strftime('%Y-%m-%d-%H-%M')}): ")
    
    if not user_timestamp:
        print("No timestamp entered. Exiting.")
        sys.exit(0)
        
    # --- Get a new config specifically for that timestamp ---
    run_config = utils.get_config(timestamp=user_timestamp)
    
    utils.setup_directories(run_config)
    run_final_validation(run_config)