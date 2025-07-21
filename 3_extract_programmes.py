# 3_extract_programmes.py
#
# Task 3: Extracts unique programme codes and names from a comma-delimited file.
# - Reads '02-bachelors-programmes-mod.csv'.
# - Selects the relevant columns.
# - Removes duplicate rows.
# - Saves the clean data to 'programmes.csv'.

import pandas as pd
import utils

def run_programme_extraction(config):
    """Performs the full programme code extraction workflow."""
    print("--- Starting Task 3: Extract Unique Programmes ---")

    # 1. Load the Comma-Delimited File (CSV)
    try:
        programmes_df = pd.read_csv(
            config.data_paths['bachelors_programmes'],
            sep=',',
            encoding='latin-1',
        )
    except FileNotFoundError as e:
        print(f"Error: Could not find input file {e.filename}.")
        print("Please create the data files first by running '0_create_dummy_data.py'.")
        return
        
    print("Loaded programmes file.")

    # 2. Select Columns of Interest
    # Check if columns exist before trying to select them
    required_cols = ['CODE', 'PROGRAMME']
    if not all(col in programmes_df.columns for col in required_cols):
        print(f"Error: Input file is missing one of the required columns: {required_cols}")
        return
        
    unique_programmes = programmes_df[required_cols]

    # 3. Remove Duplicate Rows
    original_rows = len(unique_programmes)
    unique_programmes = unique_programmes.drop_duplicates().reset_index(drop=True)
    deduplicated_rows = len(unique_programmes)
    print(f"Removed {original_rows - deduplicated_rows} duplicate rows.")

    # 4. Save the Output
    unique_programmes.to_csv(config.output_paths['programmes'], index=False)
    print(f"Unique programmes saved to {config.output_paths['programmes']}")
    print("--- Task 3 Complete ---")

if __name__ == "__main__":
    # --- Get the Configuration for the Active Session ---
    config = utils.get_config()
    print(f"--- Running Programme Extraction for Session: {config.session_name} ---")
    run_programme_extraction(config)