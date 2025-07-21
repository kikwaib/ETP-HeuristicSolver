# 7_heuristic_solver.py
#
# The main heuristic timetabling engine.
# STEP 1: Building the core Timetable class.

import pandas as pd
import random
import math
import sys
from tqdm import tqdm
import collections
from itertools import combinations
import utils
import glob
# import os

class Timetable:
    """
    A class to manage the state and evaluation of a timetable solution.
    This is the core data structure for the heuristic solver.
    """
    
    def __init__(self, exams_df, rooms_df, conflicts_df, enrollments_df, config):
        self.config = config
        # --- CORRECTED LINE ---
        self.params = config.heuristic_params 
        
        self.weights = self.params['penalty_weights']
        
        # --- Store data as dictionaries for fast lookups ---
        self.exams = exams_df.set_index('ExamCode').to_dict('index')
        self.rooms_list = rooms_df.sort_values(by='Capacity', ascending=False).to_dict('records')
        self.enrollments = enrollments_df
        
        # --- Pre-process conflicts into an efficient lookup structure ---
        self.conflicts = collections.defaultdict(int)
        for _, row in conflicts_df.iterrows():
            key = tuple(sorted((row['ExamCode1'], row['ExamCode2'])))
            self.conflicts[key] = row['ConflictStrength']
            
        # --- Pre-process student schedules for fast penalty calculations ---
        self.student_schedules = self.enrollments.groupby('StudentReg')['CourseCode'].apply(set).to_dict()

        # --- The main state of the timetable ---
        self.schedule = {}
        self.slot_to_exams = collections.defaultdict(set)
        self.slot_to_rooms = collections.defaultdict(set)
        
        # --- Cache for the total score to avoid re-calculation ---
        self.cached_score = None

    def find_valid_placements(self, exam_id):
        """
        Generator function that yields all valid (slot, rooms) assignments for a given exam.
        This checks hard constraints only: concurrency and capacity.
        """
        exam_info = self.exams[exam_id]
        
        # Check for cross-listing concurrency
        required_slot = None
        if pd.notna(exam_info['CrosslistedID']) and exam_info['CrosslistedID'] != '':
            for e, (slot, _) in self.schedule.items():
                if self.exams[e]['CrosslistedID'] == exam_info['CrosslistedID']:
                    required_slot = slot
                    break

        slots_to_check = [required_slot] if required_slot else [f"{d} {p}" for d in self.config.problem_params['days'] for p in self.config.problem_params['periods_per_day']]

        for slot in slots_to_check:
            # Find available rooms in this slot
            used_rooms = self.slot_to_rooms.get(slot, set())
            available_rooms = [r for r in self.rooms_list if r['RoomID'] not in used_rooms]

            # --- Logic for placing the exam in rooms ---
            # Case 1: Exam fits in a single available room
            for room in available_rooms:
                if exam_info['TotalEnrollment'] <= room['Capacity']:
                    yield (slot, [room['RoomID']])

            # Case 2: Large Exam Splitting - needs multiple rooms
            # Use a greedy approach: fill largest available rooms first
            if exam_info['TotalEnrollment'] > available_rooms[0]['Capacity'] if available_rooms else 0:
                rooms_for_split = []
                capacity_so_far = 0
                for room in available_rooms: # Already sorted by capacity desc
                    rooms_for_split.append(room['RoomID'])
                    capacity_so_far += room['Capacity']
                    if capacity_so_far >= exam_info['TotalEnrollment']:
                        yield (slot, rooms_for_split)
                        break

    def apply_move(self, exam_id, new_slot, new_rooms):
        """Applies a change to the schedule state."""
        # 1. Remove the exam from its old assignment (if it exists)
        if exam_id in self.schedule:
            old_slot, old_rooms = self.schedule[exam_id]
            self.slot_to_exams[old_slot].remove(exam_id)
            for room in old_rooms:
                # Check if this room is now empty in this slot
                is_room_still_used = any(room in r for e, (s, r) in self.schedule.items() if s == old_slot and e != exam_id)
                if not is_room_still_used:
                    self.slot_to_rooms[old_slot].remove(room)

        # 2. Add the exam to its new assignment
        self.schedule[exam_id] = (new_slot, new_rooms)
        self.slot_to_exams[new_slot].add(exam_id)
        for room in new_rooms:
            self.slot_to_rooms[new_slot].add(room)

        # 3. Invalidate the score cache
        self.cached_score = None

    def get_total_score(self):
        """Calculates the total weighted penalty score for the entire schedule."""
        if self.cached_score is not None:
            return self.cached_score

        total_penalty = 0
        
        # Helper structures for efficient calculation
        student_daily_exams = collections.defaultdict(lambda: collections.defaultdict(int))
        student_daily_slots = collections.defaultdict(lambda: collections.defaultdict(set))
        
        for student, courses in self.student_schedules.items():
            for exam, (slot, _) in self.schedule.items():
                if exam in courses:
                    day, period = slot.split(' ')
                    student_daily_exams[student][day] += 1
                    student_daily_slots[student][day].add(period)

        # 1. Student Conflict Penalty
        if self.params['optimize_student_conflicts']:
            conflict_penalty = 0
            for slot, exams_in_slot in self.slot_to_exams.items():
                if len(exams_in_slot) > 1:
                    for exam1, exam2 in combinations(exams_in_slot, 2):
                        key = tuple(sorted((exam1, exam2)))
                        conflict_penalty += self.conflicts.get(key, 0)
            total_penalty += conflict_penalty * self.weights['student_conflict']

        # 2. Back-to-Back and Daily Load Penalties
        b2b_penalty = 0
        load_penalty = 0
        if self.params['optimize_back_to_back'] or self.params['optimize_daily_load']:
            periods_ordered = self.config.problem_params['periods_per_day']
            for student, days in student_daily_slots.items():
                for day, scheduled_periods in days.items():
                    # Daily load
                    if self.params['optimize_daily_load']:
                        num_exams = len(scheduled_periods)
                        if num_exams > self.params['max_exams_per_day']:
                            load_penalty += (num_exams - self.params['max_exams_per_day'])
                    # Back-to-back
                    if self.params['optimize_back_to_back'] and len(scheduled_periods) > 1:
                        for i in range(len(periods_ordered) - 1):
                            if periods_ordered[i] in scheduled_periods and periods_ordered[i+1] in scheduled_periods:
                                b2b_penalty += 1
            total_penalty += b2b_penalty * self.weights['back_to_back']
            total_penalty += load_penalty * self.weights['daily_load']

        # 3. Midday Preference Penalty
        if self.params['optimize_midday_preference']:
            pref_penalty = 0
            midday_periods = set(self.config.problem_params['midday_periods'])
            for exam, (slot, _) in self.schedule.items():
                if self.exams[exam]['MPreferred'] and slot.split(' ')[1] not in midday_periods:
                    pref_penalty += 1
            total_penalty += pref_penalty * self.weights['midday_preference']
            
        # 4. Room Usage and Packing Penalty
        if self.params['optimize_room_packing']:
            room_penalty = 0
            for slot, exams_in_slot in self.slot_to_exams.items():
                rooms_in_slot = self.slot_to_rooms[slot]
                room_contents = collections.defaultdict(list)
                for exam in exams_in_slot:
                    exam_rooms = self.schedule[exam][1]
                    for room in exam_rooms:
                        room_contents[room].append(exam)
                
                room_penalty += len(rooms_in_slot) * self.weights['room_usage']
                
                packing_level = self.params['packing_level']
                if packing_level != "none":
                    grouping_key = 'SchName' if packing_level == 'school' else 'DeptName'
                    mix_weight = self.weights['school_mixing'] if packing_level == 'school' else self.weights['department_mixing']
                    for room, contents in room_contents.items():
                        if len(contents) > 1:
                            unique_groups = set(self.exams[ex][grouping_key] for ex in contents)
                            if len(unique_groups) > 1:
                                room_penalty += (len(unique_groups) - 1) * mix_weight
            total_penalty += room_penalty

        self.cached_score = total_penalty
        return total_penalty

    def calculate_marginal_penalty(self, exam_id, slot, rooms):
        """
        EFFICIENTLY calculates the increase in penalty score if a given
        exam is placed at a specific slot and rooms, given the current schedule.
        This function calculates only the DELTA, not the full score.
        """
        marginal_penalty = 0
        day, period = slot.split(' ')
        exam_info = self.exams[exam_id]

        # 1. Student Conflict Penalty
        if self.params['optimize_student_conflicts']:
            exams_in_slot = self.slot_to_exams.get(slot, set())
            for other_exam in exams_in_slot:
                key = tuple(sorted((exam_id, other_exam)))
                marginal_penalty += self.conflicts.get(key, 0) * self.weights['student_conflict']
        
        # 2. Back-to-Back and Daily Load Penalties
        if self.params['optimize_back_to_back'] or self.params['optimize_daily_load']:
            students_in_exam = self.get_students_for_exam(exam_id)
            periods_ordered = self.config.problem_params['periods_per_day']
            period_index = periods_ordered.index(period)

            for student in students_in_exam:
                exams_on_day = self.get_student_exams_on_day(student, day)
                
                # Daily Load Delta
                if self.params['optimize_daily_load']:
                    if len(exams_on_day) >= self.params['max_exams_per_day']:
                        marginal_penalty += self.weights['daily_load']
                
                # Back-to-Back Delta
                if self.params['optimize_back_to_back']:
                    # Check for conflict with exam in previous period
                    if period_index > 0:
                        prev_period = periods_ordered[period_index - 1]
                        if any(self.schedule[e][0] == f"{day} {prev_period}" for e in exams_on_day):
                            marginal_penalty += self.weights['back_to_back']
                    # Check for conflict with exam in next period
                    if period_index < len(periods_ordered) - 1:
                        next_period = periods_ordered[period_index + 1]
                        if any(self.schedule[e][0] == f"{day} {next_period}" for e in exams_on_day):
                            marginal_penalty += self.weights['back_to_back']

        # 3. Midday Preference Penalty
        if self.params['optimize_midday_preference']:
            if exam_info['MPreferred'] and period not in self.config.problem_params['midday_periods']:
                marginal_penalty += self.weights['midday_preference']

        # 4. Room Usage and Packing Penalty
        if self.params['optimize_room_packing']:
            # Base cost for any NEW rooms being used in this slot
            used_rooms_in_slot = self.slot_to_rooms.get(slot, set())
            for room in rooms:
                if room not in used_rooms_in_slot:
                    marginal_penalty += self.weights['room_usage']
            
            # Mixing cost if we are packing into an existing room
            if len(rooms) == 1:
                room_id = rooms[0]
                exams_in_room = [e for e, (s, r_list) in self.schedule.items() if s == slot and room_id in r_list]
                if exams_in_room: # We are packing
                    packing_level = self.params['packing_level']
                    if packing_level != 'none':
                        grouping_key = 'SchName' if packing_level == 'school' else 'DeptName'
                        mix_weight = self.weights['school_mixing'] if packing_level == 'school' else self.weights['department_mixing']
                        
                        existing_groups = {self.exams[e][grouping_key] for e in exams_in_room}
                        if exam_info[grouping_key] not in existing_groups:
                            marginal_penalty += mix_weight
        return marginal_penalty
        
    def remove_exam(self, exam_id):
        """Helper to remove an exam from the schedule."""
        if exam_id in self.schedule:
            old_slot, old_rooms = self.schedule.pop(exam_id)
            self.slot_to_exams[old_slot].remove(exam_id)
            for room in old_rooms:
                is_room_still_used = any(room in r for e, (s, r) in self.schedule.items() if s == old_slot and e != exam_id)
                if not is_room_still_used:
                    self.slot_to_rooms[old_slot].remove(room)
            self.cached_score = None

    def to_dataframe(self):
        """Converts the current schedule state into a pandas DataFrame."""
        schedule_list = []
        for exam, (slot, rooms) in self.schedule.items():
            day, period = slot.split(' ')
            # If an exam is in multiple rooms (split), create a row for each room
            for room in rooms:
                schedule_list.append({'ExamCode': exam, 'Day': day, 'Period': period, 'RoomID': room})
        return pd.DataFrame(schedule_list)

    def get_students_for_exam(self, exam_id):
        """Helper to get all students enrolled in a specific exam."""
        # This is a placeholder for a potentially more efficient lookup if needed.
        # For now, we query the enrollments DataFrame.
        return set(self.enrollments[self.enrollments['CourseCode'] == exam_id]['StudentReg'])

    def get_student_exams_on_day(self, student_id, day):
        """Helper to find which exams a student is taking on a specific day."""
        student_courses = self.student_schedules.get(student_id, set())
        exams_on_day = set()
        for exam in student_courses:
            if exam in self.schedule and self.schedule[exam][0].startswith(day):
                exams_on_day.add(exam)
        return exams_on_day
    
# --- Greedy Construction Algorithm ---
def greedy_construction(data, config):
    """Builds an initial high-quality solution by placing the hardest exams first."""
    print("\n--- Phase 1: Greedy Construction ---")
    
    timetable = Timetable(
        data['exam_master_list'], data['rooms'],
        data['exam_conflict_graph'], data['enrollments'], config
    )

    # Calculate a difficulty score for each exam
    exam_difficulty = collections.defaultdict(int)
    for exam_id, exam_info in timetable.exams.items():
        exam_difficulty[exam_id] += exam_info['TotalEnrollment']
    
    for (e1, e2), strength in timetable.conflicts.items():
        exam_difficulty[e1] += strength
        exam_difficulty[e2] += strength
        
    sorted_exams = sorted(exam_difficulty.keys(), key=lambda e: exam_difficulty[e], reverse=True)
    
    unplaced_exams = []
    
    for exam_id in tqdm(sorted_exams, desc="Greedy Placement"):
        min_penalty = float('inf')
        best_assignment = None
        
        for slot, rooms in timetable.find_valid_placements(exam_id):
            current_penalty = timetable.calculate_marginal_penalty(exam_id, slot, rooms)
            if current_penalty < min_penalty:
                min_penalty = current_penalty
                best_assignment = (slot, rooms)

        if best_assignment:
            timetable.apply_move(exam_id, best_assignment[0], best_assignment[1])
        else:
            # If no valid placement was found, add it to our list
            unplaced_exams.append(exam_id)
            print(f"\nCRITICAL WARNING: Could not find any valid placement for exam {exam_id}. The problem might be infeasible.", file=sys.stderr)

    print("Greedy construction complete.")

    save_params = config.heuristic_params['intermediate_saves']
    if save_params['enabled']:
        output_dir = config.output_paths['final_schedule']
        save_path = f"{save_params['save_directory']}/00_initial_greedy_solution.csv"
        timetable.to_dataframe().to_csv(save_path, index=False)
        print(f"Initial greedy solution saved to: {save_path}")

    return timetable, unplaced_exams

# --- Phase 2: Improvement via Simulated Annealing ---
def simulated_annealing(initial_timetable, data):
    """
    Improves the initial solution using a Simulated Annealing metaheuristic.
    Saves the best-found solution intermittently.
    """
    print("\n--- Phase 2: Simulated Annealing Improvement ---")
    
    config = initial_timetable.config
    params = initial_timetable.params
    save_params = params['intermediate_saves']
    
    current_timetable = initial_timetable
    
    current_score = current_timetable.get_total_score()
    best_score = current_score
    
    # To save the best solution, we need to store its state (a simple dictionary)
    best_schedule_state = current_timetable.schedule.copy()
    
    temp = params['sa_initial_temperature']
    min_temp = params['sa_min_temperature']
    cooling_rate = params['sa_cooling_rate']
    
     # --- Only choose from exams that are ACTUALLY in the schedule ---
    all_exams = list(current_timetable.schedule.keys())
    improvement_counter = 0

    with tqdm(total=int(temp), desc="Simulated Annealing", unit=" temp") as pbar:
        while temp > min_temp:
            pbar.update(int(pbar.total - temp) - pbar.n)
            
            for _ in range(params['sa_iterations_per_temp']):
                exam_to_move = random.choice(all_exams)
                original_slot, original_rooms = current_timetable.schedule[exam_to_move]
                
                possible_placements = list(current_timetable.find_valid_placements(exam_to_move))
                if not possible_placements: continue

                new_slot, new_rooms = random.choice(possible_placements)
                
                # Apply, score, and decide
                current_timetable.apply_move(exam_to_move, new_slot, new_rooms)
                new_score = current_timetable.get_total_score()
                delta_score = new_score - current_score
                
                if delta_score < 0: # Good move
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_schedule_state = current_timetable.schedule.copy()
                        improvement_counter += 1
                        pbar.set_postfix_str(f"Best Score: {best_score:,.0f}")
                        
                        # --- NEW: Save the new best solution ---
                        if save_params['enabled']:
                            output_dir = config.output_paths['final_schedule']
                            save_path = f"{save_params['save_directory']}/{improvement_counter:04d}_solution_score_{best_score:,.0f}.csv"

                            # Create a temporary timetable object with the best state to save it
                            temp_best_tt = Timetable(data['exam_master_list'], data['rooms'], data['exam_conflict_graph'], data['enrollments'], config)
                            temp_best_tt.schedule = best_schedule_state
                            temp_best_tt.to_dataframe().to_csv(save_path, index=False)

                elif math.exp(-delta_score / temp) > random.random(): # Accept bad move
                    current_score = new_score
                else: # Reject bad move
                    current_timetable.apply_move(exam_to_move, original_slot, original_rooms)
            
            temp *= cooling_rate
            
    print("Simulated annealing complete.")
    
    # Before returning, ensure the final timetable object reflects the best state found
    final_timetable = Timetable(data['exam_master_list'], data['rooms'], data['exam_conflict_graph'], data['enrollments'], config)
    final_timetable.schedule = best_schedule_state
    
    return final_timetable

# --- Main execution logic ---
def main():
    """Main execution function. Creates a new run with a new timestamp."""
    
    # --- 1. Get a new Configuration for this run ---
    # Calling get_config() without a timestamp generates a new one.
    config = utils.get_config()
    print(f"--- Starting New Solver Run ---")
    print(f"Session: {config.session_name}")
    print(f"Run ID (Timestamp): {config.run_timestamp}")
    
    # --- 2. Run the main logic ---
    utils.setup_directories(config)

    # --- Load all necessary data files ---
    print("--- Loading All Input Data for Heuristic Solver ---")
    try:
        data = {
            'exam_master_list': pd.read_csv(config.output_paths['exam_master_list']),
            'rooms': pd.read_csv(config.data_paths['rooms']),
            'exam_conflict_graph': pd.read_csv(config.output_paths['exam_conflict_graph']),
        }
        # Load all enrollment files using glob
        enrollment_files = glob.glob(config.data_paths['enrollment_files_pattern'])
        df_list = [pd.read_csv(f, header=None, encoding='latin1', usecols=[1, 5]) for f in enrollment_files]
        raw_enrollments_df = pd.concat(df_list, ignore_index=True)
        raw_enrollments_df.columns = ['StudentReg', 'CourseCode']
        data['enrollments'] = raw_enrollments_df
    except FileNotFoundError as e:
        print(f"ERROR: A required input file is missing: {e.filename}", file=sys.stderr)
        print("Please ensure all data preparation scripts (1-5) have been run successfully.", file=sys.stderr)
        sys.exit(1)

    # Phase 1: Greedy Construction
    initial_solution, unplaced_exams = greedy_construction(data, config)
    print(f"\nInitial greedy solution found with score: {initial_solution.get_total_score():,.2f}")
    
    # --- Save the unplaced exams report ---
    if unplaced_exams:
        unplaced_df = data['exam_master_list'][data['exam_master_list']['ExamCode'].isin(unplaced_exams)]
        unplaced_df = unplaced_df[['ExamCode', 'TotalEnrollment']]
        output_path = config.output_paths['unplaced_exams_report'] # Uses timestamped path
        unplaced_df.to_csv(output_path, index=False)
        print(f"\nReport of {len(unplaced_df)} unplaced exams saved to: {output_path}")

    # Phase 2: Improvement
    final_solution = simulated_annealing(initial_solution, data)
    final_score = final_solution.get_total_score()
    print(f"\nFinal optimized solution found with score: {final_score:,.2f}")

    # --- Save final outputs with the run's timestamp ---
    final_schedule_df = final_solution.to_dataframe()
    final_schedule_path = config.output_paths['final_schedule'] # Uses timestamped path
    final_schedule_df.to_csv(final_schedule_path, index=False)
    print(f"\nFinal schedule saved to {final_schedule_path}")
    
    print("\n--- Solver Run Finished ---")
    print(f"To generate validation reports for this run, use the timestamp: {config.run_timestamp}")

    # generate_validation_report(final_solution) # Placeholder for the final validation script

if __name__ == "__main__":
    main()