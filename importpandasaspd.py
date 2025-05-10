import pandas as pd
import numpy as np
import random

# --- Script to Analyze Original Dataset Patterns ---

# Load the original dataset
try:
    df_original = pd.read_csv('Emergency_Department_Staffing_Optimization_Dataset (1).csv')
except FileNotFoundError:
    print("Error: 'Emergency_Department_Staffing_Optimization_Dataset (1).csv' not found.")
    print("Please ensure the file is in the same directory as this script, or provide the full path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# Convert 'datetime' column to datetime objects
df_original['datetime'] = pd.to_datetime(df_original['datetime'])

# Extract day of week name and hour of day
df_original['day_of_week_name'] = df_original['datetime'].dt.day_name()
df_original['hour_of_day'] = df_original['datetime'].dt.hour

# Define the order for days of the week for more readable output
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_original['day_of_week_name'] = pd.Categorical(df_original['day_of_week_name'], categories=days_order, ordered=True)

print("--- Analyzing Temporal Patterns in Original Dataset ---")

# Calculate and display average patients_arrived
try:
    avg_arrivals = df_original.groupby(['day_of_week_name', 'hour_of_day'], observed=False)['patients_arrived'].mean().unstack()
    print("\nAverage Patients Arrived (by Day of Week and Hour):")
    print(avg_arrivals)
except KeyError:
    print("\nError: 'patients_arrived' column not found in the dataset.")
except Exception as e:
    print(f"\nAn error occurred during arrival analysis: {e}")


# Calculate and display average avg_acuity_level
try:
    avg_acuity = df_original.groupby(['day_of_week_name', 'hour_of_day'], observed=False)['avg_acuity_level'].mean().unstack()
    print("\nAverage Acuity Level (by Day of Week and Hour):")
    print(avg_acuity)
except KeyError:
    print("\nError: 'avg_acuity_level' column not found in the dataset.")
except Exception as e:
    print(f"\nAn error occurred during acuity analysis: {e}")


print("\n--- How to Interpret These Patterns ---")
print("Examine the tables above for 'patients_arrived' and 'avg_acuity_level'.")
print("Typical Emergency Department (ED) variations include:")
print("  - Peaks: Often on Monday mornings, weekday evenings (e.g., after work hours),")
print("           and sometimes surges during weekends.")
print("  - Lulls: Typically during overnight hours (e.g., 1 AM - 6 AM), and potentially")
print("           quieter periods mid-day on certain weekdays or parts of the weekend.")
print("If the values in your tables appear relatively consistent across different hours and days")
print("(i.e., they look 'flat'), the base dataset might not fully reflect these common ED patterns.")

print("\n--- Visualization Suggestion (for a Jupyter notebook or similar environment) ---")
print("To better visualize these patterns, you can use heatmaps. Here's example code:")
print("""
# import matplotlib.pyplot as plt
# import seaborn as sns

# if 'avg_arrivals' in locals() and not avg_arrivals.empty:
#   plt.figure(figsize=(14, 7))
#   sns.heatmap(avg_arrivals, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
#   plt.title('Average Patients Arrived by Day of Week and Hour')
#   plt.ylabel('Day of Week')
#   plt.xlabel('Hour of Day')
#   plt.tight_layout()
#   plt.show()
# else:
#   print("Average arrivals data is not available for plotting.")

# if 'avg_acuity' in locals() and not avg_acuity.empty:
#   plt.figure(figsize=(14, 7))
#   sns.heatmap(avg_acuity, annot=True, fmt=".1f", cmap="OrRd", linewidths=.5)
#   plt.title('Average Acuity Level by Day of Week and Hour')
#   plt.ylabel('Day of Week')
#   plt.xlabel('Hour of Day')
#   plt.tight_layout()
#   plt.show()
# else:
#   print("Average acuity data is not available for plotting.")
""")
print("--- End of Analysis Script ---")

# Load your existing CSV
df = pd.read_csv('Emergency_Department_Staffing_Optimization_Dataset (1).csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# --- Overlay More Realistic Hourly/Daily Arrival Patterns ---
# Define multipliers for hour of day (0-23) - customize as needed
hourly_arrival_multipliers = {
    0: 0.5, 1: 0.4, 2: 0.3, 3: 0.3, 4: 0.4, 5: 0.5, # Overnight lull
    6: 0.7, 7: 0.9, 8: 1.1, 9: 1.3, 10: 1.4, 11: 1.3, # Morning ramp-up & peak
    12: 1.2, 13: 1.1, 14: 1.1, 15: 1.2, 16: 1.3, # Afternoon
    17: 1.4, 18: 1.5, 19: 1.4, 20: 1.2, 21: 1.0, # Evening surge
    22: 0.8, 23: 0.6  # Late evening decline
}
# Define multipliers for day of week (Monday=0, Sunday=6) - customize as needed
daily_arrival_multipliers = {
    0: 1.25, # Monday (e.g., busiest)
    1: 1.15, # Tuesday
    2: 1.00, # Wednesday (baseline)
    3: 1.05, # Thursday
    4: 1.20, # Friday (evening surge might start earlier or be more pronounced)
    5: 1.10, # Saturday (can vary)
    6: 1.15  # Sunday (often busy, especially afternoon/evening)
}

df['hour_of_day_temp'] = df['datetime'].dt.hour
df['day_of_week_num_temp'] = df['datetime'].dt.dayofweek

# Apply multipliers to 'patients_arrived'
df['patients_arrived'] = df.apply(
    lambda row: row['patients_arrived'] * \
                hourly_arrival_multipliers.get(row['hour_of_day_temp'], 1.0) * \
                daily_arrival_multipliers.get(row['day_of_week_num_temp'], 1.0),
    axis=1
)
# Ensure patients_arrived remains integer and non-negative
df['patients_arrived'] = np.maximum(0, df['patients_arrived'].round().astype(int))

# Clean up temporary columns used for this adjustment
del df['hour_of_day_temp']
del df['day_of_week_num_temp']
# --- End of Pattern Overlay ---

# --- Define Hospital Characteristics ---
hospital_configs = {
    'H1': {'type': 'Safety-Net', 'icu_beds': 10, 'base_patient_multiplier': 1.2, 'staff_multiplier': 0.8},
    'H2': {'type': 'Private', 'icu_beds': 20, 'base_patient_multiplier': 0.9, 'staff_multiplier': 1.1},
    'H3': {'type': 'Academic', 'icu_beds': 15, 'base_patient_multiplier': 1.0, 'staff_multiplier': 1.0}
}
hospital_ids = list(hospital_configs.keys())
num_hospitals = len(hospital_ids)
num_rows_original = len(df)

# --- Create a larger dataset by replicating for each hospital ---
all_dfs = []
for i, hosp_id in enumerate(hospital_ids):
    hosp_df = df.copy()
    hosp_df['hospital_id'] = hosp_id
    hosp_df['hospital_type'] = hospital_configs[hosp_id]['type']
    all_dfs.append(hosp_df)

df_expanded = pd.concat(all_dfs, ignore_index=True)

# --- Add New Columns & Generate Synthetic Values ---
df_expanded['icu_beds_total'] = df_expanded['hospital_id'].apply(lambda x: hospital_configs[x]['icu_beds'])

# Initialize new columns
df_expanded['icu_occupancy_pct'] = 0.0
df_expanded['diagnostic_imaging_avg_wait_time_min'] = 0.0
df_expanded['critical_equipment_down_pct_time'] = 0.0
df_expanded['sep1_compliance_pct'] = 0.0
df_expanded['readmissions_within_30_days'] = 0

# --- Initialize ICU Occupancy for First Entry of Each Hospital ---
# This ensures a realistic baseline before the main loop processes this row.
unique_hospital_ids = df_expanded['hospital_id'].unique()
for hosp_id_init in unique_hospital_ids:
    # Find the index of the first data row for this hospital
    first_occurrence_indices = df_expanded[df_expanded['hospital_id'] == hosp_id_init].index
    if not first_occurrence_indices.empty:
        first_index_for_hospital = first_occurrence_indices[0]
        # Set icu_occupancy_pct directly to a random float between 50.0 and 80.0
        df_expanded.loc[first_index_for_hospital, 'icu_occupancy_pct'] = random.uniform(50.0, 80.0)
# --- End of ICU Initialization ---

# --- Make existing metrics more dynamic and add logic for new ones ---
for index, row in df_expanded.iterrows():
    # Basic load factor
    load_factor = (row['patients_arrived'] * row['avg_acuity_level']) / \
                  ((row['doctors_on_shift'] + row['nurses_on_shift'] * 0.5) + 1e-6) # Avoid division by zero
    
    h_type = row['hospital_type']
    h_config = hospital_configs[row['hospital_id']]

    # Critical Equipment Down (small chance)
    df_expanded.loc[index, 'critical_equipment_down_pct_time'] = np.random.choice([0.0, 0.05, 0.1], p=[0.95, 0.04, 0.01]) * random.uniform(0.5, 1)
    
    # Diagnostic Imaging Wait Time
    base_diag_wait = 15 + 5 * load_factor
    if h_type == 'Safety-Net':
        base_diag_wait *= 1.3
    elif h_type == 'Private':
        base_diag_wait *= 0.8
    # Significantly increase wait time if equipment is down
    base_diag_wait *= (1 + df_expanded.loc[index, 'critical_equipment_down_pct_time'] * 15) # Increased multiplier
    df_expanded.loc[index, 'diagnostic_imaging_avg_wait_time_min'] = max(5, np.random.normal(base_diag_wait, 5))

    # Arrival to Provider Time
    # Adjusted to be more sensitive to load_factor and calculated diagnostic_imaging_avg_wait_time_min
    base_atp = 10 + 10 * load_factor # Base ATP driven by load factor
    base_atp += df_expanded.loc[index, 'diagnostic_imaging_avg_wait_time_min'] * 0.2 # Increased sensitivity
    if h_type == 'Safety-Net':
        base_atp *= 1.15 # Slightly increased multiplier
    df_expanded.loc[index, 'arrival_to_provider_time_min'] = max(5, min(150, np.random.normal(base_atp, base_atp*0.15))) # Increased cap and variance

    # Left Without Being Seen (LWBS)
    # Adjusted for higher sensitivity to load_factor and calculated arrival_to_provider_time_min
    base_lwbs = 0.0
    if load_factor > 2.0: # Lowered threshold
         base_lwbs = (load_factor - 2.0) * 3.0 # Increased multiplier
    
    calculated_arrival_to_provider_time = df_expanded.loc[index, 'arrival_to_provider_time_min']
    if calculated_arrival_to_provider_time > 25: # Lowered threshold
         base_lwbs += (calculated_arrival_to_provider_time - 25) * 0.15 # Increased multiplier
    if h_type == 'Safety-Net':
        base_lwbs *= 1.6 # Increased multiplier
    elif h_type == 'Private':
        base_lwbs *= 0.7
    df_expanded.loc[index, 'left_without_being_seen_pct'] = max(0, min(25, np.random.normal(base_lwbs, 2) + row['left_without_being_seen_pct']*0.2))

    # ED Length of Stay (ED LOS)
    # Adjusted for higher sensitivity to load_factor
    base_ed_los = 60 + 25 * load_factor + row['avg_acuity_level'] * 10 # Increased load_factor multiplier
    base_ed_los += df_expanded.loc[index, 'diagnostic_imaging_avg_wait_time_min'] * 0.5
    # Amplify LOS if critical equipment is down
    base_ed_los *= (1 + df_expanded.loc[index, 'critical_equipment_down_pct_time'] * 0.75) # Added impact
    if row['admissions'] > 0:
        base_ed_los += 60 * row['admissions']
    if h_type == 'Safety-Net':
        base_ed_los *= 1.2
    elif h_type == 'Private':
        base_ed_los *= 0.9
    df_expanded.loc[index, 'ed_length_of_stay_min'] = max(30, min(720, np.random.normal(base_ed_los, 30)))

    # ICU Occupancy
    icu_demand_from_admissions = row['admissions'] * (row['avg_acuity_level'] / 5.0) * 0.3
    prev_index = index - 1
    # Determine if this is the first data row for this particular hospital_id
    is_first_row_for_this_hospital = (index == 0) or (df_expanded.loc[prev_index, 'hospital_id'] != row['hospital_id'])

    if is_first_row_for_this_hospital:
        # For the first row of this hospital, 'icu_occupancy_pct' was pre-initialized.
        # This pre-initialized value is the occupancy percentage at the START of this hour.
        initial_occupancy_percentage_at_start_of_hour = df_expanded.loc[index, 'icu_occupancy_pct']
        current_icu_beds_used_at_start_of_hour = df_expanded.loc[index, 'icu_beds_total'] * (initial_occupancy_percentage_at_start_of_hour / 100.0)
    else:
        # For subsequent rows, base current beds used on the *previous hour's final* occupancy percentage.
        current_icu_beds_used_at_start_of_hour = df_expanded.loc[index, 'icu_beds_total'] * (df_expanded.loc[prev_index, 'icu_occupancy_pct'] / 100.0)
    
    # Calculate new_icu_beds_used by adding admissions and subtracting discharges for the *current* hour
    new_icu_beds_used_at_end_of_hour = current_icu_beds_used_at_start_of_hour + icu_demand_from_admissions - (random.uniform(0,2) * h_config['icu_beds']/10.0)
    # Calculate and set the final icu_occupancy_pct for the *current* hour
    df_expanded.loc[index, 'icu_occupancy_pct'] = max(0, min(100, (new_icu_beds_used_at_end_of_hour / (df_expanded.loc[index, 'icu_beds_total'] + 1e-6)) * 100))

    # SEP-1 Compliance
    # Adjusted for higher sensitivity to load_factor and calculated arrival_to_provider_time_min
    base_sep1 = 90
    base_sep1 -= load_factor * 8 # Increased multiplier
    
    calculated_arrival_to_provider_time_for_sep1 = df_expanded.loc[index, 'arrival_to_provider_time_min']
    base_sep1 -= (calculated_arrival_to_provider_time_for_sep1 / 7) # Increased impact
    base_sep1 -= (row['avg_acuity_level'] - 3) * 5
    # Amplify compliance drop if critical equipment is down
    base_sep1 -= df_expanded.loc[index, 'critical_equipment_down_pct_time'] * 100 # Increased multiplier
    if h_type == 'Safety-Net':
        base_sep1 -= 10
    elif h_type == 'Private':
        base_sep1 += 5
    df_expanded.loc[index, 'sep1_compliance_pct'] = max(30, min(100, np.random.normal(base_sep1, 5)))

    # Patient Satisfaction
    # Adjusted for higher sensitivity to load_factor and other calculated KPIs
    base_satisfaction = 95
    base_satisfaction -= load_factor * 3 # Added direct load_factor impact
    base_satisfaction -= df_expanded.loc[index, 'left_without_being_seen_pct'] * 0.8 # Increased multiplier
    
    calculated_arrival_to_provider_time_for_pss = df_expanded.loc[index, 'arrival_to_provider_time_min']
    base_satisfaction -= (calculated_arrival_to_provider_time_for_pss / 3) # Increased impact
    
    calculated_ed_los_for_pss = df_expanded.loc[index, 'ed_length_of_stay_min']
    base_satisfaction -= (calculated_ed_los_for_pss / 15) # Increased impact
    if df_expanded.loc[index, 'icu_occupancy_pct'] > 85: # Lowered threshold for ICU full impact
        base_satisfaction -= 7 # Increased impact
    if h_type == 'Safety-Net':
        base_satisfaction -= 5
    elif h_type == 'Private':
        base_satisfaction += 3
    df_expanded.loc[index, 'patient_satisfaction_score'] = max(50, min(100, np.random.normal(base_satisfaction, 3)))

    # Readmissions within 30 days
    # Adjusted probabilities to aim for a slightly higher overall rate (e.g., 4-6%)
    base_readmit_prob = 0.015  # Increased baseline from 0.01
    if row['admissions'] > 0:
        base_readmit_prob += 0.02  # Kept as is
    if row['avg_acuity_level'] > 4.0:
        base_readmit_prob += 0.02  # Slightly increased from 0.015
    if df_expanded.loc[index, 'ed_length_of_stay_min'] > 360: 
        base_readmit_prob += 0.015 # Increased from 0.01
    if h_type == 'Safety-Net':
        base_readmit_prob += 0.01  # Increased from 0.005
    # Add a small bonus for Academic hospitals if acuity is high (to differentiate slightly)
    if h_type == 'Academic' and row['avg_acuity_level'] > 4.2:
        base_readmit_prob += 0.005
    # Add a small bonus for Private hospitals if ED LOS is very high (to differentiate slightly)
    if h_type == 'Private' and df_expanded.loc[index, 'ed_length_of_stay_min'] > 400:
        base_readmit_prob += 0.005

    final_readmit_prob = min(base_readmit_prob, 0.10) # Cap probability remains 10%

    if random.random() < final_readmit_prob:
        df_expanded.loc[index, 'readmissions_within_30_days'] = 1
    else:
        df_expanded.loc[index, 'readmissions_within_30_days'] = 0
    
# Ensure LWBS is not > 100
df_expanded['left_without_being_seen_pct'] = df_expanded['left_without_being_seen_pct'].clip(0, 100)

# Fill any remaining NaNs (e.g. from initial ICU occupancy logic for first row of a hospital)
# Fill NaNs that might occur if the first row of a hospital group couldn't calculate ICU occupancy based on a previous row
df_expanded['icu_occupancy_pct'] = df_expanded.groupby('hospital_id')['icu_occupancy_pct'].bfill().ffill()


print("Generated synthetic data. Here's a sample:")
print(df_expanded.head())
print("\nInfo:")
df_expanded.info()
print("\nDescribe:")
print(df_expanded.describe())

# Save the updated dataset
df_expanded.to_csv('Emergency_Department_Staffing_Optimization_Dataset_Augmented_V3.csv', index=False)
print("\nSuccessfully saved augmented data to Emergency_Department_Staffing_Optimization_Dataset_Augmented_V3.csv")