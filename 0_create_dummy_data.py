# 0_create_dummy_data.py
#
# UPDATED to use the new session-based structure.
# This script will always generate data in the session folder specified by
# SESSION_NAME in utils.py (which should be '00-dummy' when running this).

import os
import pandas as pd
import utils

def create_dummy_data(config):
    """Generates a consistent set of sample CSV files for the session."""
    utils.setup_directories(config)
    print(f"--- Generating Dummy Data in '{config['data_dir']}' ---")

    # --- 1. Enrollment Data ---
    enrollment_data = [
        ['', 'SCS-001-2023', '', 'SCS-2023', '', 'CSC101'], ['', 'SCS-001-2023', '', 'SCS-2023', '', 'MTH101'],
        ['', 'SCS-001-2023', '', 'SCS-2023', '', 'PHY101'], ['', 'SCS-002-2023', '', 'SCS-2023', '', 'CSC101'],
        ['', 'SCS-002-2023', '', 'SCS-2023', '', 'MTH101'], ['', 'SCS-002-2023', '', 'SCS-2023', '', 'ENG110'],
        ['', 'SCS-003-2023', '', 'SCS-2023', '', 'CSC101'], ['', 'SCS-003-2023', '', 'SCS-2023', '', 'MTH202'],
        ['', 'SCS-003-2023', '', 'SCS-2023', '', 'MUS100'], ['', 'SPH-101-2023', '', 'SPH-2023', '', 'PHY101'],
        ['', 'SPH-101-2023', '', 'SPH-2023', '', 'MTH101'], ['', 'SPH-101-2023', '', 'SPH-2023', '', 'CSC101'],
        ['', 'SPH-102-2023', '', 'SPH-2023', '', 'PHY101'], ['', 'SPH-102-2023', '', 'SPH-2023', '', 'MTH101'],
        ['', 'SPH-102-2023', '', 'SPH-2023', '', 'MTH202'], ['', 'BHI-501-2022', '', 'BHI-2022', '', 'HST101'],
        ['', 'BHI-501-2022', '', 'BHI-2022', '', 'ENG110'], ['', 'BHI-502-2022', '', 'BHI-2022', '', 'HST101'],
        ['', 'BHI-502-2022', '', 'BHI-2022', '', 'ENG110'], ['', 'BHI-502-2022', '', 'BHI-2022', '', 'HST250'],
        ['', 'BHI-503-2022', '', 'BHI-2022', '', 'HST250'], ['', 'BHI-503-2022', '', 'BHI-2022', '', 'ENG320'],
    ]
    # concatenate config['data_dir'] and enrollment_part1.csv to create file.
    pd.DataFrame(enrollment_data[:11]).to_csv(os.path.join(config['data_dir'], 'enrollment_part1.csv'), header=False, index=False)
    pd.DataFrame(enrollment_data[11:]).to_csv(os.path.join(config['data_dir'], 'enrollment_part2.csv'), header=False, index=False)
    print("Generated enrollment_part1.csv and enrollment_part2.csv")

    # --- 2. Course Details File ---
    units_courses_data = [
        ['CSC101', 'Introduction to Programming', 'Computer Science'], ['CSC202', 'Data Structures', 'Computer Science'],
        ['CSC305', 'Artificial Intelligence', 'Computer Science'], ['MTH101', 'Calculus I', 'Mathematics'],
        ['MTH202', 'Linear Algebra', 'mathematics'], ['PHY101', 'Classical Mechanics', 'Physics'],
        ['HST101', 'World History I', 'History'], ['HST250', 'Modern European History', 'History'],
        ['ENG110', 'Academic Writing', 'English'], ['ENG320', 'Shakespearean Literature', 'English'],
        ['MUS100', 'Intro to Music Theory', 'Music'],
    ]
    pd.DataFrame(units_courses_data).to_csv(os.path.join(config['data_dir'], 'all_units_courses.csv'), header=False, index=False)
    print(f"Generated {config['data_dir']}/all_units_courses.csv")

    # --- 3. Departments File ---
    departments_data = {
        'Dept Code': ['CSC', 'MTH', 'PHY', 'HST', 'ENG', 'MUS'],
        'Dept Name': ['Computer Science', 'Mathematics', 'Physics', 'History', 'English', 'Music'],
        'School Code': ['SCS', 'SCS', 'SPS', 'SOH', 'SOH', 'SFA'],
        'School Name': ['School of Computing', 'School of Computing', 'School of Physical Sciences', 'School of Humanities', 'School of Humanities', 'School of Fine Arts'],
        'MPreferred': [True, True, True, False, False, False]
    }
    pd.DataFrame(departments_data).to_csv(os.path.join(config['data_dir'], 'departments.csv'), index=False)
    print(f"Generated {os.path.join(config['data_dir'], 'departments.csv')}")

    # --- 4. Rooms and Capacities File ---
    rooms_data = {
        'RoomID': [
            'LectureHall_A', 'LectureHall_B', 'Auditorium_C', 'Classroom_101',
            'Classroom_102', 'Classroom_201', 'Classroom_202', 'Lab_A',
            'Lab_B', 'Seminar_Room_1', 'Seminar_Room_2'
        ],
        'Capacity': [
            500, 350, 200, 100, 100, 75, 75, 50, 50, 25, 25
        ]
    }
    pd.DataFrame(rooms_data).to_csv(os.path.join(config['data_dir'], 'rooms.csv'), index=False)
    print(f"Generated {os.path.join(config['data_dir'], 'rooms.csv')}")

    # --- 5. Programmes File (TSV with duplicates) ---
    programmes_data = {
        'CODE': ['SCS', 'SPH', 'BHI', 'SCS', 'SPH', 'BHI', 'SED', 'SCS', 'SPH', 'BHI', 'SEC'],
        'PROGRAMME': ['BSc. Computer Science', 'BSc. Physics', 'BA. History', 'BSc. Computer Science', 'BSc. Physics', 'BA. History', 'BEd. Science', 'BSc. Computer Science', 'BSc. Physics', 'BA. History', 'BSc. Economics'],
        'OTHER COL A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'OTHER COL B': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    }
    pd.DataFrame(programmes_data).to_csv(os.path.join(config['data_dir'], 'bachelors_programmes.csv'), index=False)
    print(f"Generated {os.path.join(config['data_dir'], 'bachelors_programmes.csv')}")
    print("\n--- Dummy data generation complete. All required files are now available. ---")

if __name__ == "__main__":
    # Get the configuration for the currently active session
    # For this script to work as intended, SESSION_NAME in utils.py should be "00-dummy"
    config = utils.get_config()
    if config['session_name'] != "00-dummy":
        print(f"Warning: It is recommended to set SESSION_NAME to '00-dummy' in utils.py before generating dummy data.")
    else:
        create_dummy_data(config)