# utils.py
#
# Shared library for the ETP Project.
# Implements a session-based directory structure.
# To switch between datasets, simply change the `SESSION_NAME` variable below.

import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import glob
from datetime import datetime

# ######################################################################## #
# ###########            MASTER CONFIGURATION SWITCH           ########### #
# ######################################################################## #
#
#  CHOOSE THE WORKING SESSION HERE.
#  All scripts will automatically use the data and output folders for this session.
#
# SESSION_NAME = "00-dummy"  # <--- CHANGE THIS VALUE
# SESSION_NAME = "JULY-2025-SUPS" # <--- Example for a real session
SESSION_NAME = "JAN-APR-2024"

# ######################################################################## #
class Config:
    """
    A configuration class to manage paths and parameters dynamically.
    Accepts an optional timestamp to either create a new run or load an existing one.
    """
    def __init__(self, session_name, timestamp=None):
        self.session_name = session_name
        
        # If no timestamp is provided, generate a new one for a new run.
        # Otherwise, use the provided timestamp to reference an existing run.
        if timestamp is None:
            self.run_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        else:
            self.run_timestamp = timestamp
        """
        Generates the configuration dictionary for the globally defined SESSION_NAME.
        """
        # Define base directories
        self.session_dir = os.path.join("sessions", self.session_name)
        self.data_dir = os.path.join(self.session_dir, "data")
        self.output_dir = os.path.join(self.session_dir, "output")

        self.data_paths = {
            "enrollment_files_pattern": os.path.join(self.data_dir, "enrollment_*.csv"),
            "all_units_courses": os.path.join(self.data_dir, "all_units_courses.csv"),
            "departments": os.path.join(self.data_dir, "departments.csv"),
            "bachelors_programmes": os.path.join(self.data_dir, "bachelors_programmes.csv"),
            "rooms": os.path.join(self.data_dir, "rooms.csv"),
        }
        self.output_paths = {
            "cluster_summary": os.path.join(self.output_dir, "cluster_summary.csv"),
            "student_cluster_assignments": os.path.join(self.output_dir, "student_cluster_assignments.csv"),
            "course_details": os.path.join(self.output_dir, "course_details.csv"),
            "programmes": os.path.join(self.output_dir, "programmes.csv"),
            "cohort_cluster_analysis": os.path.join(self.output_dir, "cohort_cluster_analysis.csv"),
            "exam_conflict_graph": os.path.join(self.output_dir, "exam_conflict_graph.csv"),
            "cluster_exam_enrollments": os.path.join(self.output_dir, "cluster_exam_enrollments.csv"),
            "exam_master_list": os.path.join(self.output_dir, "exam_master_list.csv"),
            "final_schedule": os.path.join(self.output_dir, "final_schedule.csv"),
            "student_conflict_report": os.path.join(self.output_dir, "student_conflict_report.csv"),
            "unplaced_exams_report": os.path.join(self.output_dir, "unplaced_exams_report.csv"),
            "validation_report": os.path.join(self.output_dir, "validation_report.txt"),
        }

        self.clustering_params = {
            # The number of student clusters (cohorts) to create.
            # This is a key parameter to tune for the timetabling model.
            "k_clusters": 100,
        }

        self.problem_params = {
            #"days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
            "days": ["W1-Mon", "W1-Tue", "W1-Wed", "W1-Thu", "W1-Fri", "W2-Mon", "W2-Tue", "W2-Wed", "W2-Thu", "W2-Fri"],
            "periods_per_day": ["08:30-10:30", "11:00-13:00", "14:00-16:00"],
            # This list must contain strings that exactly match entries in 'periods_per_day'.
            "midday_periods": ["11:00-13:00"],
        }

        self.heuristic_params = {
            # --- GENERAL ---
            # Defines the administrative level for the packing mixing penalty.
            # Options: "school", "department", "none".
            # "school": Penalizes mixing exams from different schools in the same room.
            # "department": Penalizes mixing exams from different departments.
            # "none": Disables the mixing penalty entirely. Room cost penalty still applies.
            "packing_level": "school", # Group packed exams by "school" or "department"
            "max_exams_per_day": 2,    # Used for the daily load penalty

            # --- CONFIGURATION FLAGS to enable/disable soft constraints ---
            "optimize_student_conflicts": True,
            "optimize_back_to_back": False,
            "optimize_daily_load": False,
            "optimize_midday_preference": False,
            "optimize_room_packing": False,

            # --- PENALTY WEIGHTS ---
            # The solver will try to minimize the sum of: Penalty * Weight
            "penalty_weights": {
                "student_conflict": 1000.0, # Per student per conflict
                "back_to_back": 50.0,       # Per student per back-to-back occurrence
                "daily_load": 200.0,      # Per student per exam over the daily max
                "midday_preference": 500.0, # Per exam
                "room_usage": 10.0,         # Per room used per slot
                "school_mixing": 250.0,     # Per extra school mixed in a packed room
                "department_mixing": 150.0, # Per extra department mixed in a packed room
            },
            
            # --- SIMULATED ANNEALING (Local Search) PARAMETERS ---
            "sa_initial_temperature": 1000.0,
            "sa_cooling_rate": 0.9995,
            "sa_min_temperature": 0.1,
            "sa_iterations_per_temp": 100, # Number of moves to try at each temperature

            # ---  INTERMEDIATE SAVE PARAMETERS ---
            "intermediate_saves": {
                "enabled": True, # Set to False to disable saving intermediate solutions
                # A subdirectory within the session's output folder to store best solutions
                "save_directory": os.path.join(self.output_dir, "best_solutions_found"),
            }
        }

def get_config(timestamp=None):
    """Instantiates and returns the configuration object for the active session."""
    return Config(SESSION_NAME, timestamp=timestamp)

def setup_directories(config):
    """Ensures that the data and output directories for the session exist."""
    print(f"--- Setting up directories for session: '{config.session_name}' ---")
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Also create the intermediate save directory if enabled
    if config.heuristic_params['intermediate_saves']['enabled']:
        os.makedirs(config.heuristic_params['intermediate_saves']['save_directory'], exist_ok=True)
        
    print(f"Directories are ready.")

def parallel_load_data(file_paths: dict, max_workers: int) -> dict:
    """Loads multiple CSV files concurrently."""
    print(f"\n--- Starting Concurrent Data Ingestion ---")
    
    all_paths = []
    # Handle glob patterns
    for key, path_or_pattern in file_paths.items():
        if "*" in path_or_pattern:
            all_paths.extend(glob.glob(path_or_pattern))
        else:
            all_paths.append(path_or_pattern)
    
    if not all_paths:
        print("Warning: No files found to load.", file=sys.stderr)
        return {}

    data_frames = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a mapping from future back to the original key and specific path
        future_to_info = {}
        for key, path_or_pattern in file_paths.items():
            if "*" in path_or_pattern:
                files = glob.glob(path_or_pattern)
                for file_path in files:
                    future = executor.submit(pd.read_csv, file_path, header=None)
                    future_to_info[future] = (key, file_path)
            else:
                future = executor.submit(pd.read_csv, path_or_pattern)
                future_to_info[future] = (key, path_or_pattern)

        # Process results as they complete
        for future in tqdm(as_completed(future_to_info), total=len(future_to_info), desc="Loading files"):
            key, path = future_to_info[future]
            try:
                df = future.result()
                if key not in data_frames:
                    data_frames[key] = []
                data_frames[key].append(df)
            except FileNotFoundError:
                print(f"\nERROR: Cannot find file {path}. Please ensure it exists.", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"\nERROR: Could not load {path}. {e}", file=sys.stderr)
                sys.exit(1)

    # Concatenate dataframes that came from the same key (glob pattern)
    final_data = {}
    for key, df_list in data_frames.items():
        final_data[key] = pd.concat(df_list, ignore_index=True)
    
    print("Data ingestion finished.")
    return final_data