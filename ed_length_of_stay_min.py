#!/usr/bin/env python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression # Added for classification option
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import joblib # Added import
import os # Added import

# 1. Load the dataset
file_path = 'Emergency_Department_Staffing_Optimization_Dataset_Augmented_V3.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Dataset file not found at {file_path}")
    exit()

print("Dataset loaded successfully.")
print(f"Shape of the dataset: {df.shape}")

# Define model save directory
MODEL_SAVE_DIR = "." # Models will be saved in the current directory

# 2. Feature Engineering from 'datetime'
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour_of_day'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek # Monday=0, Sunday=6
df['is_weekend'] = df['day_of_week'].isin([5, 6])

print("\nTime-based features created.")

# 3. Define common feature columns
feature_columns = [
    'patients_arrived',
    'avg_acuity_level',
    'doctors_on_shift',
    'nurses_on_shift',
    'hospital_type',
    'icu_beds_total',
    'icu_occupancy_pct',
    'critical_equipment_down_pct_time',
    'hour_of_day',
    'day_of_week',
    'is_weekend'
]

# Ensure categorical features are of type 'category' for OneHotEncoder before the loop
df['hour_of_day'] = df['hour_of_day'].astype('category')
df['day_of_week'] = df['day_of_week'].astype('category')
df['is_weekend'] = df['is_weekend'].astype('category')
df['hospital_type'] = df['hospital_type'].astype('category')

# --- Step 1.4: Generalizing KPI Model Training and Storage ---
print("\n--- Starting Generalized KPI Model Training ---")

trained_models = {}
kpi_preprocessors = {}

kpi_configs = {
    'arrival_to_provider_time_min': {'type': 'regression', 'model_class': RandomForestRegressor, 'params': {'random_state': 42, 'n_jobs': -1, 'n_estimators': 100}},
    'ed_length_of_stay_min': {'type': 'regression', 'model_class': RandomForestRegressor, 'params': {'random_state': 42, 'n_jobs': -1, 'n_estimators': 100}},
    'left_without_being_seen_pct': {'type': 'regression', 'model_class': RandomForestRegressor, 'params': {'random_state': 42, 'n_jobs': -1, 'n_estimators': 100}},
    'sep1_compliance_pct': {'type': 'regression', 'model_class': RandomForestRegressor, 'params': {'random_state': 42, 'n_jobs': -1, 'n_estimators': 100}},
    'patient_satisfaction_score': {'type': 'regression', 'model_class': RandomForestRegressor, 'params': {'random_state': 42, 'n_jobs': -1, 'n_estimators': 100}},
    'readmissions_within_30_days': {'type': 'classification', 'model_class': RandomForestClassifier, 'params': {'random_state': 42, 'n_jobs': -1, 'n_estimators': 100}} # Using RandomForestClassifier as requested
}

# Define categorical and numerical features (common for all models)
categorical_features = ['hospital_type', 'hour_of_day', 'day_of_week', 'is_weekend']
numerical_features = [col for col in feature_columns if col not in categorical_features]

for kpi_name, config in kpi_configs.items():
    print(f"\n--- Processing KPI: {kpi_name} ({config['type']}) ---")

    # a. Define X and y for the current KPI
    X = df[feature_columns] # Use a fresh slice from the original df
    y = df[kpi_name]

    # b. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # c. Create and fit the appropriate preprocessing pipeline on the training data
    # The preprocessor definition can be the same, but it must be FIT on each KPI's X_train
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    print(f"Fitting preprocessor for {kpi_name}...")
    X_train_prepared = preprocessor.fit_transform(X_train.copy()) # Fit on current KPI's X_train
    X_test_prepared = preprocessor.transform(X_test.copy())     # Transform current KPI's X_test
    print(f"Shape of preprocessed training features: {X_train_prepared.shape}")

    # d. Train the chosen model type
    print(f"Training {config['model_class'].__name__} for {kpi_name}...")
    model = config['model_class'](**config['params'])
    model.fit(X_train_prepared, y_train)
    print("Model training complete.")

    # e. Store the fitted model and its preprocessor
    trained_models[kpi_name] = model
    kpi_preprocessors[kpi_name] = preprocessor # Store the fitted preprocessor

    # g. Save the fitted model and preprocessor to disk
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Create directory if it doesn't exist
    model_path = os.path.join(MODEL_SAVE_DIR, f"{kpi_name}_model.joblib")
    preprocessor_path = os.path.join(MODEL_SAVE_DIR, f"{kpi_name}_preprocessor.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"  Saved model to {model_path}")
    print(f"  Saved preprocessor to {preprocessor_path}")

    # f. Print the evaluation metrics for the test set
    print(f"\nEvaluating model for {kpi_name} on the test set...")
    y_pred = model.predict(X_test_prepared)

    if config['type'] == 'regression':
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"  R-squared: {r2:.4f}")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    elif config['type'] == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        # For ROC AUC, predict_proba might be needed for some models like LogisticRegression
        # RandomForestClassifier has predict_proba by default.
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_prepared)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"  ROC AUC: {roc_auc:.4f}")
        else: # Fallback for models without predict_proba, though ROC AUC is less meaningful then
            try:
                roc_auc = roc_auc_score(y_test, y_pred)
                print(f"  ROC AUC (from predict): {roc_auc:.4f}")
            except ValueError as e:
                print(f"  ROC AUC could not be calculated: {e}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")

print("\n--- Generalized KPI Model Training Complete ---")
print(f"\nTrained models stored in 'trained_models' dictionary: {list(trained_models.keys())}")
print(f"Fitted preprocessors stored in 'kpi_preprocessors' dictionary: {list(kpi_preprocessors.keys())}")

# Example: How to access a model and its preprocessor
# if 'ed_length_of_stay_min' in trained_models:
#     example_model = trained_models['ed_length_of_stay_min']
#     example_preprocessor = kpi_preprocessors['ed_length_of_stay_min']
#     print(f"\nExample: Retrieved model for ed_length_of_stay_min: {example_model}")
#     print(f"Example: Retrieved preprocessor for ed_length_of_stay_min: {example_preprocessor}")

# --- Phase 2: Resource Allocation Strategy Recommendation ---

def recommend_optimal_staffing(
    input_scenario_features_original: pd.Series, # Expecting a pandas Series for easier manipulation
    kpi_models_dict: dict, 
    kpi_preprocessors_dict: dict, # Pass preprocessors separately
    kpi_configs_dict: dict, # To know KPI type (regression/classification) and if proba is needed
    doctor_adjustment_range: range, 
    nurse_adjustment_range: range, 
    min_doctors: int = 1, 
    min_nurses: int = 1
):
    """
    Recommends optimal staffing adjustments based on predicting KPI outcomes.

    Args:
        input_scenario_features_original: Pandas Series with current situation features.
        kpi_models_dict: Dictionary of trained KPI models.
        kpi_preprocessors_dict: Dictionary of fitted preprocessors for each KPI.
        kpi_configs_dict: Configuration for each KPI (type, model_class, etc.).
        doctor_adjustment_range: Range of adjustments for doctors.
        nurse_adjustment_range: Range of adjustments for nurses.
        min_doctors: Minimum number of doctors.
        min_nurses: Minimum number of nurses.

    Returns:
        A dictionary with recommended staffing and predicted KPIs.
    """
    
    current_doctors = input_scenario_features_original['doctors_on_shift']
    current_nurses = input_scenario_features_original['nurses_on_shift']

    best_strategy = {
        'doctors': current_doctors,
        'nurses': current_nurses,
        'predicted_lwbs': float('inf'), # Primary goal: minimize LWBS
        'total_staff': float('inf'),    # Secondary goal: minimize staff count
        'predicted_kpis': {}
    }
    
    evaluated_strategies = []

    # --- Function to predict KPIs for a given scenario --- 
    def get_predicted_kpis(scenario_df, models, preprocessors, configs):
        predicted_kpis_output = {}
        for kpi_name, model in models.items():
            # Ensure scenario_df is a DataFrame for the preprocessor
            # Preprocessor expects categorical features to be of 'category' dtype
            # This should ideally be handled before this function if possible, 
            # or ensure dtypes match training time.
            
            # Create a fresh copy for each KPI to avoid issues with dtype conversions if any happen in preprocessor
            scenario_features_for_kpi = scenario_df.copy()
            
            # Ensure categorical columns are set to 'category' dtype as expected by preprocessor
            # This is a bit of a safeguard; ideally, dtypes are consistent.
            if 'hour_of_day' in scenario_features_for_kpi.columns: scenario_features_for_kpi['hour_of_day'] = scenario_features_for_kpi['hour_of_day'].astype('category')
            if 'day_of_week' in scenario_features_for_kpi.columns: scenario_features_for_kpi['day_of_week'] = scenario_features_for_kpi['day_of_week'].astype('category')
            if 'is_weekend' in scenario_features_for_kpi.columns: scenario_features_for_kpi['is_weekend'] = scenario_features_for_kpi['is_weekend'].astype('category')
            if 'hospital_type' in scenario_features_for_kpi.columns: scenario_features_for_kpi['hospital_type'] = scenario_features_for_kpi['hospital_type'].astype('category')
            
            preprocessor_kpi = preprocessors[kpi_name]
            prepared_scenario = preprocessor_kpi.transform(scenario_features_for_kpi)
            
            if configs[kpi_name]['type'] == 'classification' and hasattr(model, "predict_proba"):
                # For classification, store probability of positive class (e.g., readmission=1)
                predicted_kpis_output[kpi_name] = model.predict_proba(prepared_scenario)[0, 1] 
            else:
                predicted_kpis_output[kpi_name] = model.predict(prepared_scenario)[0]
        return predicted_kpis_output

    # --- Evaluate baseline (current staffing) --- 
    # Convert Series to DataFrame for preprocessor compatibility
    baseline_scenario_df = input_scenario_features_original.to_frame().T 
    baseline_predicted_kpis = get_predicted_kpis(
        baseline_scenario_df, 
        kpi_models_dict, 
        kpi_preprocessors_dict, 
        kpi_configs_dict
    )
    print(f"Baseline KPIs (Current Staffing: {current_doctors} Docs, {current_nurses} Nurses): {baseline_predicted_kpis}")
    
    # Update best_strategy if baseline LWBS is relevant for comparison initially
    if 'left_without_being_seen_pct' in baseline_predicted_kpis:
        best_strategy['predicted_lwbs'] = baseline_predicted_kpis['left_without_being_seen_pct']
        best_strategy['total_staff'] = current_doctors + current_nurses
        best_strategy['predicted_kpis'] = baseline_predicted_kpis

    # --- Iterate through staffing strategy space ---
    print(f"\nExploring staffing strategies around current: {current_doctors} Docs, {current_nurses} Nurses...")
    for doc_adj in doctor_adjustment_range:
        new_doctors = current_doctors + doc_adj
        if new_doctors < min_doctors: continue

        for nurse_adj in nurse_adjustment_range:
            new_nurses = current_nurses + nurse_adj
            if new_nurses < min_nurses: continue

            # Create a copy of the input scenario to modify staffing
            modified_scenario_series = input_scenario_features_original.copy()
            modified_scenario_series['doctors_on_shift'] = new_doctors
            modified_scenario_series['nurses_on_shift'] = new_nurses
            
            # Convert to DataFrame for preprocessor compatibility
            modified_scenario_df = modified_scenario_series.to_frame().T
            
            current_strategy_kpis = get_predicted_kpis(
                modified_scenario_df, 
                kpi_models_dict, 
                kpi_preprocessors_dict, 
                kpi_configs_dict
            )
            
            current_lwbs = current_strategy_kpis.get('left_without_being_seen_pct', float('inf'))
            current_total_staff = new_doctors + new_nurses
            
            evaluated_strategies.append({
                'doctors': new_doctors,
                'nurses': new_nurses,
                'predicted_lwbs': current_lwbs,
                'total_staff': current_total_staff,
                'predicted_kpis': current_strategy_kpis
            })

            # d. Compare strategies
            if current_lwbs < best_strategy['predicted_lwbs']:
                best_strategy.update({
                    'doctors': new_doctors,
                    'nurses': new_nurses,
                    'predicted_lwbs': current_lwbs,
                    'total_staff': current_total_staff,
                    'predicted_kpis': current_strategy_kpis
                })
            elif current_lwbs == best_strategy['predicted_lwbs']:
                if current_total_staff < best_strategy['total_staff']:
                    best_strategy.update({
                        'doctors': new_doctors,
                        'nurses': new_nurses,
                        'predicted_lwbs': current_lwbs,
                        'total_staff': current_total_staff,
                        'predicted_kpis': current_strategy_kpis
                    })
    
    print(f"Explored {len(evaluated_strategies)} staffing strategies.")

    return {
        'recommended_doctors': best_strategy['doctors'],
        'recommended_nurses': best_strategy['nurses'],
        'predicted_kpis_for_recommendation': best_strategy['predicted_kpis'],
        'baseline_predicted_kpis': baseline_predicted_kpis
    }

# The kpi_configs_dict should be available from the previous training step.
# If not running in the same session, it needs to be redefined or loaded.
# For now, let's redefine it as it was in the training script for completeness if testing this part standalone.
if 'kpi_configs' not in locals(): # Check if kpi_configs exists from previous script execution
    print("Redefining kpi_configs for recommendation engine testing (if run standalone)")
    kpi_configs = {
        'arrival_to_provider_time_min': {'type': 'regression', 'model_class': RandomForestRegressor},
        'ed_length_of_stay_min': {'type': 'regression', 'model_class': RandomForestRegressor},
        'left_without_being_seen_pct': {'type': 'regression', 'model_class': RandomForestRegressor},
        'sep1_compliance_pct': {'type': 'regression', 'model_class': RandomForestRegressor},
        'patient_satisfaction_score': {'type': 'regression', 'model_class': RandomForestRegressor},
        'readmissions_within_30_days': {'type': 'classification', 'model_class': RandomForestClassifier}
    }

# Note: The dictionaries 'trained_models' and 'kpi_preprocessors' 
# (populated in Step 1.4) are expected to be available in the global scope 
# when this function is called after running the training script part.

print("\n--- Recommendation Engine Function Defined: recommend_optimal_staffing ---")
print("To test, call recommend_optimal_staffing() with a sample scenario and the trained models/preprocessors.")

# --- Step 2.3: Testing the Recommendation Engine ---
if __name__ == '__main__': # Ensure this test runs only when script is executed directly
    print("\n\n--- Testing the Recommendation Engine ---")
    
    # Check if all necessary variables from model training are available
    required_vars_exist = (
        'df' in locals() and not df.empty and 
        'trained_models' in locals() and trained_models and 
        'kpi_preprocessors' in locals() and kpi_preprocessors and 
        'kpi_configs' in locals() and kpi_configs
    )

    if required_vars_exist:
        # 1. Select a sample row for the input scenario
        # Ensure all feature_columns are in df before sampling
        if not all(col in df.columns for col in feature_columns):
            print("Error: Not all expected feature_columns are present in the base DataFrame 'df'.")
            print(f"Expected: {feature_columns}")
            print(f"Actual df columns: {df.columns.tolist()}")
        else:
            sample_scenario_original_row = df[feature_columns].sample(1, random_state=42).iloc[0]
            print("\nSelected Sample Scenario (Original Features):")
            print(sample_scenario_original_row)
            print(f"Original Staffing: Docs={sample_scenario_original_row['doctors_on_shift']}, Nurses={sample_scenario_original_row['nurses_on_shift']}")
            
            print("Original KPIs from sample row (if available in source data for this specific row - note these are NOT predictions):")
            original_kpis_to_show = [
                'arrival_to_provider_time_min', 'ed_length_of_stay_min', 
                'left_without_being_seen_pct', 'sep1_compliance_pct', 
                'patient_satisfaction_score', 'readmissions_within_30_days'
            ]
            for kpi_key in original_kpis_to_show:
                if kpi_key in df.columns:
                    original_kpi_val = df.loc[sample_scenario_original_row.name, kpi_key]
                    print(f"  {kpi_key}: {original_kpi_val}")
                else:
                    print(f"  {kpi_key}: (Not directly available in source features for this scenario display)")

            doc_adj_range = range(-2, 3)
            nurse_adj_range = range(-3, 4)
            min_docs_test = 1
            min_nurses_test = 1

            recommendation_output = recommend_optimal_staffing(
                input_scenario_features_original=sample_scenario_original_row,
                kpi_models_dict=trained_models,
                kpi_preprocessors_dict=kpi_preprocessors,
                kpi_configs_dict=kpi_configs,
                doctor_adjustment_range=doc_adj_range,
                nurse_adjustment_range=nurse_adj_range,
                min_doctors=min_docs_test,
                min_nurses=min_nurses_test
            )

            print("\n--- Recommendation Results ---")
            print(f"Original Staffing from Scenario: Doctors={sample_scenario_original_row['doctors_on_shift']}, Nurses={sample_scenario_original_row['nurses_on_shift']}")
            
            print("\nBaseline Predicted KPIs (for original staffing):")
            for kpi, value in recommendation_output['baseline_predicted_kpis'].items():
                print(f"  Predicted {kpi}: {value:.4f}")
            
            print(f"\nRecommended Staffing: Doctors={recommendation_output['recommended_doctors']}, Nurses={recommendation_output['recommended_nurses']}")
            print("Predicted KPIs for Recommended Staffing:")
            for kpi, value in recommendation_output['predicted_kpis_for_recommendation'].items():
                print(f"  Predicted {kpi}: {value:.4f}")
    else:
        print("Skipping Recommendation Engine test because necessary variables (df, models, preprocessors, configs) are not available.")
        print("Please ensure the full script (including data loading and model training) is run before this test.")

    print("\n--- End of Recommendation Engine Test ---")