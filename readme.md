# ETP-HeuristicSolver

> A scalable university examination timetabling solver using k-means clustering for student grouping, greedy construction, and simulated annealing optimization. Overcomes MIP scalability limitations through multi-stage decomposition.

## Overview

This project provides a complete, multi-stage framework for solving the University Examination Timetabling Problem (ETP) using a scalable, heuristic-driven approach. Instead of a monolithic Mixed-Integer Programming (MIP) model that often fails to scale to real-world instances, this solution decomposes the problem using:

- **K-means clustering** for intelligent student grouping based on enrollment patterns
- **Greedy construction** algorithm for initial feasible solution generation  
- **Simulated annealing** metaheuristic for iterative solution improvement

The primary goal of the solver is to produce a high-quality timetable that minimizes a weighted combination of penalties, with a key focus on minimizing the number of students affected by scheduling conflicts while ensuring scalability for large university instances.

## Project Structure

The project is organized into a series of sequential Python scripts:

-   `utils.py`: A shared library containing all configuration paths and parameters.
-   `0_create_dummy_data.py`: (Optional) A helper script to generate a full set of sample data.
-   **Data Preparation Scripts (1-5):** These scripts perform data cleaning, analysis, and pre-calculation to prepare the necessary inputs for the main solver.
-   **Solver Script (6):** The main heuristic engine that generates the timetable.
-   **Validation Script (7):** A final script that analyzes the output schedule and generates reports.

## Installation

The project requires Python 3 and a few common data science libraries.

1.  **Clone or download the project files** into a single directory.
2.  **Install the required libraries** using pip:
    ```bash
    pip install pandas scikit-learn tqdm
    ```

## Execution Workflow

The scripts are designed to be run sequentially from your terminal. Each step generates output files that are required by the next step.

### Step 0: (Optional) Generate Dummy Data

If you are running the project for the first time or want to test it with a sample dataset, run this script. It will create a `/data/` directory with all the necessary input files.

```bash
python 0_create_dummy_data.py
```

---

### Step 1: Data Preparation and Analysis

This sequence of scripts takes the raw institutional data and transforms it into the structured inputs required by the heuristic solver.

**1.1 Student Clustering (`1_student_clustering.py`)**
*   **Purpose:** To analyze student enrollment patterns and group students into cohorts (clusters) using the k-means algorithm.
*   **Requires:** Raw enrollment CSVs in the `data/` folder.
*   **Executes:**
    ```bash
    python 1_student_clustering.py
    ```
*   **Generates:**
    *   `output/cluster_summary.csv`
    *   `output/student_cluster_assignments.csv`

**1.2 Merge Course Details (`2_merge_course_details.py`)**
*   **Purpose:** To combine course, department, and school information into a single, clean file.
*   **Requires:** `data/01-all-units-courses.csv` and `data/departments-mod-2.csv`.
*   **Executes:**
    ```bash
    python 2_merge_course_details.py
    ```
*   **Generates:**
    *   `output/course_details.csv` (with the new `MPreferred` and blank `CrosslistedID` columns).

**1.3 Extract Programmes (`3_extract_programmes.py`)**
*   **Purpose:** To extract unique programme codes from the raw programme data file.
*   **Requires:** `data/02-bachelors-programmes.tsv`.
*   **Executes:**
    ```bash
    python 3_extract_programmes.py
    ```
*   **Generates:**
    *   `output/programmes.csv`

**1.4 Analyze Cohort Clusters (`4_analyze_cohort_clusters.py`)**
*   **Purpose:** To provide a high-level summary of the diversity of academic paths within each official student cohort.
*   **Requires:** `output/student_cluster_assignments.csv`.
*   **Executes:**
    ```bash
    python 4_analyze_cohort_clusters.py
    ```
*   **Generates:**
    *   `output/cohort_cluster_analysis.csv`

**1.5 Prepare Final Timetabling Inputs (`5_prepare_timetabling_inputs.py`)**
*   **Purpose:** This is the final and most critical preparation step. It pre-calculates the conflict graph and creates the definitive master list of exams for the solver.
*   **Requires:** Raw enrollment data, `output/student_cluster_assignments.csv`, and `output/course_details.csv`.
*   **Executes:**
    ```bash
    python 5_prepare_timetabling_inputs.py
    ```
*   **Generates:**
    *   `output/exam_conflict_graph.csv`
    *   `output/cluster_exam_enrollments.csv`
    *   `output/exam_master_list.csv`

---

### Step 2: Run the Heuristic Solver

This is the main event. This script takes the prepared data and runs the two-phase heuristic algorithm to generate the final schedule.

**2.1 Run Solver (`7_heuristic_solver.py`)**
*   **Purpose:** To construct and improve a timetable that minimizes the weighted penalty score defined in the configuration.
*   **Requires:** `output/exam_master_list.csv`, `output/exam_conflict_graph.csv`, `data/rooms.csv`, and raw enrollment data.
*   **Executes:**
    ```bash
    python 7_heuristic_solver.py
    ```
*   **Generates:**
    *   `output/final_schedule.csv`

---

### Step 3: Generate Final Reports and Validation

This final script analyzes the output from the solver to produce the definitive conflict report and a summary of hard constraint validation.

**3.1 Generate Conflict Report (`6_generate_conflict_report.py`)**
*   **Purpose:** To analyze the final schedule and produce a detailed list of any students who are still affected by conflicts, and to validate that all hard constraints were met.
*   **Requires:** `output/final_schedule.csv`, raw enrollment data, and other prepared files for context.
*   **Executes:**
    ```bash
    python 6_generate_conflict_report.py
    ```
*   **Generates:**
    *   `output/student_conflict_report.csv` (This file will be empty if the schedule is clash-free).
    *   `output/validation_report.txt`

## Configuration

All key parameters for the project can be configured in the `utils.py` file within the `CONFIG` dictionary. This includes file paths, time periods, and all parameters and weights for the heuristic solver. This allows for easy experimentation without modifying the core logic of the scripts.