#!/usr/bin/env python
import dash
import dash_bootstrap_components as dbc # Import Dash Bootstrap Components
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go # Needed for secondary y-axis
from plotly.subplots import make_subplots # Needed for secondary y-axis
import pandas as pd
from datetime import date, timedelta
import numpy as np
import joblib # Using joblib for model persistence is recommended

# --- ML Model Related Imports (Ensure these are installed) ---
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Initialize the Dash app
UNBOUNDED_FONT = "https://fonts.googleapis.com/css2?family=Unbounded:wght@400;700&display=swap"
LATO_FONT = "https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap"
ROBOTO_FONT = "https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME, UNBOUNDED_FONT, LATO_FONT, ROBOTO_FONT])
app.title = "ShiftSmart: AI for Smarter Hospital Staffing"

# 2. Load the dataset and basic feature engineering
DATA_FILE_PATH = 'Emergency_Department_Staffing_Optimization_Dataset_Augmented_V3.csv'
df = pd.DataFrame() # Initialize df as an empty DataFrame
try:
    df_temp = pd.read_csv(DATA_FILE_PATH)
    df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])
    df_temp['date'] = df_temp['datetime'].dt.date
    df_temp['hour_of_day'] = df_temp['datetime'].dt.hour
    df_temp['day_of_week_num'] = df_temp['datetime'].dt.dayofweek
    day_map = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
    df_temp['day_of_week_name'] = df_temp['day_of_week_num'].map(day_map)
    df = df_temp # Assign to global df if loading and processing is successful
    print(f"Successfully loaded and preprocessed {DATA_FILE_PATH}")
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATA_FILE_PATH}.")
except Exception as e:
    print(f"An error occurred while loading or processing the CSV: {e}")

# --- Load Trained Models, Preprocessors, and Configs --- 
# !! Placeholder: Replace this section with actual loading logic (e.g., joblib.load) !!
# This assumes you have run the training script (`ed_length_of_stay_min.py`)
# and saved the `trained_models`, `kpi_preprocessors`, and `kpi_configs` objects.
print("--- Attempting to load ML models and preprocessors (Placeholder) ---")
trained_models = {}
kpi_preprocessors = {}
# Define kpi_configs as it was during training
kpi_configs = {
    'arrival_to_provider_time_min': {'type': 'regression', 'model_class': RandomForestRegressor},
    'ed_length_of_stay_min': {'type': 'regression', 'model_class': RandomForestRegressor},
    'left_without_being_seen_pct': {'type': 'regression', 'model_class': RandomForestRegressor},
    'sep1_compliance_pct': {'type': 'regression', 'model_class': RandomForestRegressor},
    'patient_satisfaction_score': {'type': 'regression', 'model_class': RandomForestRegressor},
    'readmissions_within_30_days': {'type': 'classification', 'model_class': RandomForestClassifier}
}
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
    'day_of_week', # Note: Use day_of_week_num (0-6) based on training script
    'is_weekend'
]
categorical_features = ['hospital_type', 'hour_of_day', 'day_of_week', 'is_weekend']
numerical_features = [col for col in feature_columns if col not in categorical_features]

MODEL_SAVE_DIR = "." # Assume models are saved in the current directory
models_loaded_successfully = True
for kpi_name in kpi_configs.keys():
    try:
        # Placeholder: Simulate loading - replace with actual joblib.load
        # trained_models[kpi_name] = joblib.load(f"{MODEL_SAVE_DIR}/{kpi_name}_model.joblib")
        # kpi_preprocessors[kpi_name] = joblib.load(f"{MODEL_SAVE_DIR}/{kpi_name}_preprocessor.joblib")
        
        # --- Start Placeholder --- 
        # Create dummy objects for demonstration if loading fails
        print(f"Placeholder: Creating dummy model/preprocessor for {kpi_name}")
        if kpi_configs[kpi_name]['type'] == 'regression':
             trained_models[kpi_name] = RandomForestRegressor(random_state=42, n_estimators=10)
        else:
             trained_models[kpi_name] = RandomForestClassifier(random_state=42, n_estimators=10)
        # Create a dummy preprocessor - IN A REAL APP, THIS MUST BE THE *FITTED* PREPROCESSOR
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        # Fit dummy preprocessor on first 10 rows of df if available
        dummy_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        if not df.empty:
             # Need day_of_week (numeric), is_weekend for fitting
             temp_df_for_fit = df.head(10).copy()
             temp_df_for_fit['day_of_week'] = temp_df_for_fit['day_of_week_num']
             temp_df_for_fit['is_weekend'] = temp_df_for_fit['day_of_week_num'].isin([5, 6])
             # Ensure categoricals are category type
             for col in categorical_features:
                 if col in temp_df_for_fit:
                      temp_df_for_fit[col] = temp_df_for_fit[col].astype('category')
             # Fit on selected features
             dummy_preprocessor.fit(temp_df_for_fit[feature_columns])
        kpi_preprocessors[kpi_name] = dummy_preprocessor

            # --- ADDED: Fit the placeholder model ---
        if kpi_name in temp_df_for_fit.columns:
                X_train_dummy_transformed = dummy_preprocessor.transform(temp_df_for_fit[feature_columns])
                y_train_dummy = temp_df_for_fit[kpi_name].copy()

                if y_train_dummy.isnull().all():
                    print(f"Warning: Target KPI '{kpi_name}' is all NaN in the first 10 rows. Dummy model for {kpi_name} cannot be fitted.")
                    trained_models[kpi_name] = None # Mark as unfitted
                elif not pd.api.types.is_numeric_dtype(y_train_dummy) and kpi_configs[kpi_name]['type'] == 'regression' and not y_train_dummy.empty:
                    print(f"Warning: Target KPI '{kpi_name}' is not numeric for regression or is empty. Dummy model for {kpi_name} cannot be fitted.")
                    trained_models[kpi_name] = None # Mark as unfitted
                elif y_train_dummy.empty:
                    print(f"Warning: Target KPI '{kpi_name}' is empty. Dummy model for {kpi_name} cannot be fitted.")
                    trained_models[kpi_name] = None # Mark as unfitted
                else:
                    if kpi_configs[kpi_name]['type'] == 'classification':
                        if y_train_dummy.isnull().any():
                            print(f"Warning: Target KPI '{kpi_name}' for classification has NaNs in the first 10 rows for dummy fitting. Filling with mode (or 0 if no mode).")
                            mode_val = y_train_dummy.mode()
                            y_train_dummy = y_train_dummy.fillna(mode_val[0] if not mode_val.empty else 0).astype(int)
                        else:
                            y_train_dummy = y_train_dummy.astype(int)

                    # Now, fit the model if it hasn't been set to None
                    if trained_models[kpi_name] is not None:
                        try:
                            trained_models[kpi_name].fit(X_train_dummy_transformed, y_train_dummy)
                            print(f"Placeholder model for {kpi_name} fitted with dummy data.")
                        except Exception as fit_e:
                            print(f"Error fitting placeholder model for {kpi_name}: {fit_e}")
                            trained_models[kpi_name] = None # Mark as unfitted on error
        else:
                print(f"Warning: Target KPI column '{kpi_name}' not found in temp_df_for_fit. Dummy model for {kpi_name} will not be fitted.")
                trained_models[kpi_name] = None # Mark as unfitted
    except FileNotFoundError:
        print(f"Error: Model or preprocessor file not found for {kpi_name}. Predictions will not work.")
        models_loaded_successfully = False
        # break # Optional: stop if any model is missing
    except Exception as e:
        print(f"Error loading model/preprocessor for {kpi_name}: {e}")
        models_loaded_successfully = False
        # break

if models_loaded_successfully and trained_models and kpi_preprocessors:
    print("--- ML Models and Preprocessors loaded (or placeholders created) successfully ---")
else:
    print("--- Failed to load all ML Models/Preprocessors. Simulator functionality may be limited. ---")

# Prepare options for dropdowns - handling case where df might be empty
hospital_id_options = []
hospital_type_options = []
day_of_week_options = []
min_date_allowed = date(2000, 1, 1) # Default placeholder dates
max_date_allowed = date(2030, 12, 31)
initial_start_date = min_date_allowed
initial_end_date = max_date_allowed

# Initialize distribution_kpi_options here so it's always defined
distribution_kpi_options = [] 

if not df.empty:
    hospital_id_options = sorted(df['hospital_id'].unique().tolist())
    hospital_type_options = sorted(df['hospital_type'].unique().tolist())
    # Ensure day_map covers all unique day_of_week_num values before mapping for options
    unique_dow_nums = sorted(df['day_of_week_num'].unique())
    day_of_week_options = [day_map[i] for i in unique_dow_nums if i in day_map]
    
    if not df['date'].empty:
        min_date_allowed = df['date'].min()
        max_date_allowed = df['date'].max()
        initial_start_date = min_date_allowed
        initial_end_date = max_date_allowed

    # Populate distribution_kpi_options if df is not empty
    # These are typically the target variables we might want to see distributions of.
    # Use keys from kpi_configs that are also present in df.columns and are numeric.
    if 'kpi_configs' in globals() and isinstance(kpi_configs, dict):
        potential_kpis_for_dist = list(kpi_configs.keys())
        distribution_kpi_options = [
            kpi for kpi in potential_kpis_for_dist 
            if kpi in df.columns and pd.api.types.is_numeric_dtype(df[kpi])
        ]
        if not distribution_kpi_options and potential_kpis_for_dist:
            print(f"Warning: None of the configured KPIs ({potential_kpis_for_dist}) found as suitable numeric columns in the DataFrame for distribution plots.")
        elif not potential_kpis_for_dist: # This case implies kpi_configs was empty
             print("Warning: kpi_configs is empty. Cannot populate distribution KPI options from it.")
    else:
        print("Warning: kpi_configs not found or not a dictionary. Distribution KPI options may be based on all numeric columns or be empty.")
        # Fallback: if kpi_configs isn't available/useful, could populate with other numeric columns, e.g.
        # distribution_kpi_options = [col for col in df.select_dtypes(include=np.number).columns if col not in ['hospital_id', 'day_of_week_num', 'hour_of_day', 'avg_acuity_level', 'patients_arrived']]
else:
        print("DataFrame is empty. Filters will have default/empty options, and distribution KPI selector will be empty.")
    # distribution_kpi_options remains [] as initialized above

# --- KPI list for Distribution Analysis Selector ---
# (The options are now defined above based on df.empty, kpi_configs and DataFrame columns)

# --- Globally Accessible KPI Display Configuration ---
PRED_KPI_MAP_CONFIG = {
    'arrival_to_provider_time_min': {"name": "Arrival to Provider", "unit": " min", "max_val": 120, "color": '#17A2B8', "lower_is_better": True, "format": ".0f", "target_value": 30, "icon_unicode": "\uf2f2"}, 
    'ed_length_of_stay_min': {"name": "ED Length of Stay", "unit": " min", "max_val": 300, "color": '#6C757D', "lower_is_better": True, "format": ".0f", "target_value": 180, "icon_unicode": "\uf017"}, 
    'left_without_being_seen_pct': {"name": "Left Without Being Seen", "unit": " %", "max_val": 20, "color": '#FFC107', "lower_is_better": True, "format": ".1f", "target_value": 2, "icon_unicode": "\uf52b"}, 
    'sep1_compliance_pct': {"name": "SEP-1 Compliance", "unit": " %", "max_val": 100, "color": '#A3E4D7', "lower_is_better": False, "format": ".1f", "target_value": 90, "icon_unicode": "\uf058"}, 
    'patient_satisfaction_score': {"name": "Patient Satisfaction", "unit": " %", "max_val": 100, "color": '#A3E4D7', "lower_is_better": False, "format": ".1f", "target_value": 85, "icon_unicode": "\uf118"}, 
    'readmissions_within_30_days': {"name": "Readmission Rate", "unit": " %", "max_val": 50, "color": '#FD7E14', "lower_is_better": True, "format": ".1f", "target_value": 15, "icon_unicode": "\uf2f1"}
}

# --- Helper function for creating KPI cards (retained) ---
def create_kpi_card(title, value_text, card_style=None):
    default_style = {
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'padding': '10px', # Reduced padding for potentially smaller cards
        'textAlign': 'center',
        'margin': '5px',
        'minWidth': '140px', # Adjusted min width
        'flex': '1'
    }
    if card_style:
        default_style.update(card_style)
        
    return html.Div([
        html.H6(title, style={'margin': '0 0 5px 0', 'fontSize': '0.8em'}), # Smaller header
        html.H5(value_text, style={'margin': '0', 'fontSize': '1.2em'}) # Adjusted value font
    ], style=default_style)

# --- Helper function for empty figures (retained) ---
def create_empty_figure(message="No data to display"):
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, text=message, showarrow=False, xref='paper', yref='paper', font=dict(size=14))
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# --- Recommendation Engine Function (Copied and adapted from training script) ---
def recommend_optimal_staffing(
    input_scenario_features_original: pd.Series, 
    kpi_models_dict: dict, # Use globally loaded models
    kpi_preprocessors_dict: dict, # Use globally loaded preprocessors
    kpi_configs_dict: dict, # Use globally defined configs
    feature_columns_list: list, # Pass the list of expected feature columns
    categorical_features_list: list, # Pass the list of categorical feature names
    doctor_adjustment_range: range, 
    nurse_adjustment_range: range, 
    min_doctors: int = 1, 
    min_nurses: int = 1
):
    """ Recommends optimal staffing (adapted for dashboard). """
    # --- Nested Helper: get_predicted_kpis (Adapted) --- 
    def get_predicted_kpis_for_scenario(scenario_df, models, preprocessors, configs):
        predicted_kpis_output = {}
        for kpi_name, model in models.items():
            if kpi_name not in preprocessors:
                print(f"Warning: Preprocessor not found for {kpi_name}. Skipping prediction.")
                continue
            preprocessor_kpi = preprocessors[kpi_name]
            try:
                # Ensure column order matches training
                scenario_features_for_kpi = scenario_df[feature_columns_list].copy()
                
                # Set categorical dtypes
                for cat_col in categorical_features_list:
                     if cat_col in scenario_features_for_kpi.columns:
                         # Use categories from the original dataframe if possible
                         if cat_col in df.columns and hasattr(df[cat_col], 'cat'):
                             scenario_features_for_kpi[cat_col] = pd.Categorical(scenario_features_for_kpi[cat_col], categories=df[cat_col].cat.categories)
                         else:
                              scenario_features_for_kpi[cat_col] = scenario_features_for_kpi[cat_col].astype('category')
                         
                prepared_scenario = preprocessor_kpi.transform(scenario_features_for_kpi)
                
                if configs[kpi_name]['type'] == 'classification' and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(prepared_scenario) # proba is for one sample: shape (1, n_classes_model_knows)
                    if 1 in model.classes_: # Assuming positive class is '1'
                        # Find the index corresponding to class '1'
                        class_1_idx = np.where(model.classes_ == 1)[0]
                        if class_1_idx.size > 0:
                            predicted_kpis_output[kpi_name] = proba[0, class_1_idx[0]]
                        else:
                            # This case should ideally not be reached if '1 in model.classes_' is true.
                            print(f"Warning: Class 1 reported in model.classes_ but not found by np.where for {kpi_name}. Setting P(1) to 0.0.")
                            predicted_kpis_output[kpi_name] = 0.0
                    else:
                        # Class '1' was not seen by the model during training or model.classes_ is unexpected.
                        # So, probability of observing class '1' is 0.
                        predicted_kpis_output[kpi_name] = 0.0
                else:
                    predicted_kpis_output[kpi_name] = model.predict(prepared_scenario)[0]
            except Exception as e:
                print(f"Error predicting {kpi_name} for scenario: {e}")
                predicted_kpis_output[kpi_name] = None # Indicate error
        return predicted_kpis_output
    # --- End Nested Helper --- 

    current_doctors = input_scenario_features_original['doctors_on_shift']
    current_nurses = input_scenario_features_original['nurses_on_shift']
    best_strategy = {
        'doctors': current_doctors, 'nurses': current_nurses,
        'predicted_lwbs': float('inf'), 'total_staff': float('inf'), 'predicted_kpis': {}
    }
    
    # --- Evaluate baseline --- 
    baseline_scenario_df = input_scenario_features_original.to_frame().T
    baseline_predicted_kpis = get_predicted_kpis_for_scenario(
        baseline_scenario_df, kpi_models_dict, kpi_preprocessors_dict, kpi_configs_dict
    )
    if baseline_predicted_kpis.get('left_without_being_seen_pct') is not None:
        best_strategy['predicted_lwbs'] = baseline_predicted_kpis['left_without_being_seen_pct']
        best_strategy['total_staff'] = current_doctors + current_nurses
        best_strategy['predicted_kpis'] = baseline_predicted_kpis

    # --- Iterate through strategy space ---
    for doc_adj in doctor_adjustment_range:
        new_doctors = current_doctors + doc_adj
        if new_doctors < min_doctors: continue
        for nurse_adj in nurse_adjustment_range:
            new_nurses = current_nurses + nurse_adj
            if new_nurses < min_nurses: continue

            modified_scenario_series = input_scenario_features_original.copy()
            modified_scenario_series['doctors_on_shift'] = new_doctors
            modified_scenario_series['nurses_on_shift'] = new_nurses
            modified_scenario_df = modified_scenario_series.to_frame().T
            
            current_strategy_kpis = get_predicted_kpis_for_scenario(
                modified_scenario_df, kpi_models_dict, kpi_preprocessors_dict, kpi_configs_dict
            )
            
            current_lwbs = current_strategy_kpis.get('left_without_being_seen_pct', float('inf'))
            current_total_staff = new_doctors + new_nurses

            # Compare strategies
            if current_lwbs < best_strategy['predicted_lwbs']:
                best_strategy.update({'doctors': new_doctors, 'nurses': new_nurses, 'predicted_lwbs': current_lwbs, 'total_staff': current_total_staff, 'predicted_kpis': current_strategy_kpis})
            elif current_lwbs == best_strategy['predicted_lwbs'] and current_total_staff < best_strategy['total_staff']:
                 best_strategy.update({'doctors': new_doctors, 'nurses': new_nurses, 'predicted_lwbs': current_lwbs, 'total_staff': current_total_staff, 'predicted_kpis': current_strategy_kpis})
    
    return {
        'recommended_doctors': best_strategy['doctors'], 'recommended_nurses': best_strategy['nurses'],
        'predicted_kpis_for_recommendation': best_strategy['predicted_kpis'], 'baseline_predicted_kpis': baseline_predicted_kpis
    }

# 3. App Layout
app.layout = dbc.Container([
    # Header Row
    dbc.Row([
        dbc.Col(html.Img(src=app.get_asset_url('shift-smart.png'), height="60px"),
                width="auto", className="ps-3 align-self-center"),
        dbc.Col(html.H2("ShiftSmart: AI for Smarter Hospital Staffing",
                        style={'color': '#34495E', 'fontFamily': 'Unbounded, sans-serif'}), 
                className="pt-2 align-self-center"),
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button([html.I(className="fas fa-hospital me-1"), "My Hospital"], id='my-hospital-button', color="light", className="me-2"),
                dbc.Button([html.I(className="fas fa-info-circle me-1"), "About"], id='about-button', color="light", className="me-2"),
                dbc.Button([html.I(className="fas fa-user-circle me-1"), "Profile"], id='user-profile-button', color="light"),
            ])
        ], width="auto", className="ms-auto align-self-center")
    ], align="center", className="mb-4 mt-2 p-2", style={'backgroundColor': '#EAECEE', 'minHeight': '90px'}),
    
    # Main Content Area (formerly Operations Dashboard tab content)
    html.Div([ 
        dbc.Row([
            # Left Panel: Scenario Inputs & Controls
            dbc.Col([
                html.H4("Scenario & Staffing Controls", className="mt-3 mb-3"),
                html.Div([
                    html.Div([html.I(className="fas fa-users me-2"), html.Span("Patients Arrived (Current Hour)")], className="mb-1"),
                    dbc.InputGroup([
                        dbc.Button("-", id="minus-patients-arrived-button", n_clicks=0, color="danger", outline=True),
                        dbc.Input(id="display-patients-arrived", value=10, type="number", disabled=True, style={'textAlign': 'center'}),
                        dbc.Button("+", id="plus-patients-arrived-button", n_clicks=0, color="success", outline=True)
                    ]),
                    dcc.Input(id="main-patients-arrived", value=10, type="number", style={'display': 'none'}),
                ], className="mb-3"),
                html.Div([
                    html.Div([html.I(className="fas fa-stethoscope me-2"), html.Span("Average Acuity Level (1-5)")], className="mb-1"),
                    dbc.InputGroup([
                        dbc.Button("-", id="minus-avg-acuity-button", n_clicks=0, color="danger", outline=True),
                        dbc.Input(id="display-avg-acuity", value=3.5, type="number", disabled=True, style={'textAlign': 'center'}, step=0.1),
                        dbc.Button("+", id="plus-avg-acuity-button", n_clicks=0, color="success", outline=True)
                    ]),
                    dcc.Input(id="main-avg-acuity", value=3.5, type="number", step=0.1, style={'display': 'none'}),
                ], className="mb-3"),
                html.Div([
                    html.Div([html.I(className="fas fa-user-md me-2"), html.Span("Current Doctors on Shift")], className="mb-1"),
                    dbc.InputGroup([
                        dbc.Button("-", id="minus-doctors-on-shift-button", n_clicks=0, color="danger", outline=True),
                        dbc.Input(id="display-doctors-on-shift", value=3, type="number", disabled=True, style={'textAlign': 'center'}),
                        dbc.Button("+", id="plus-doctors-on-shift-button", n_clicks=0, color="success", outline=True)
                    ]),
                    dcc.Input(id="main-doctors-on-shift", value=3, type="number", min=1, style={'display': 'none'}),
                ], className="mb-3"),
                html.Div([
                    html.Div([html.I(className="fas fa-user-nurse me-2"), html.Span("Current Nurses on Shift")], className="mb-1"),
                    dbc.InputGroup([
                        dbc.Button("-", id="minus-nurses-on-shift-button", n_clicks=0, color="danger", outline=True),
                        dbc.Input(id="display-nurses-on-shift", value=6, type="number", disabled=True, style={'textAlign': 'center'}),
                        dbc.Button("+", id="plus-nurses-on-shift-button", n_clicks=0, color="success", outline=True)
                    ]),
                    dcc.Input(id="main-nurses-on-shift", value=6, type="number", min=1, style={'display': 'none'}),
                ], className="mb-3"),
                dbc.Label("Day of Week:"),
                dbc.Select(
                    id='main-day-of-week-selector',
                    options=[{'label': name, 'value': num} for num, name in day_map.items()],
                    value='0', 
                    className="mb-3"
                ),
                dbc.Label("Hour of Day (0-23):"),
                dcc.Slider(id='main-hour-of-day-slider', min=0, max=23, step=1, marks={i: str(i) for i in range(0, 24, 3)}, value=10, tooltip={"placement": "bottom", "always_visible": True}, className="mb-4"),
                dbc.Button('Get Staffing Recommendation', id='main-recommend-staffing-button', 
                           style={'backgroundColor': '#A3E4D7', 'color': '#212529', 'borderColor': '#A3E4D7'},
                           className="w-100 mb-3")
            ], md=3, style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'height': 'calc(100vh - 90px - 1rem)', 'overflowY': 'auto'}),
            
            dbc.Col([
                html.H5("Live KPI Overview", className="mt-3 mb-3 text-center"),
                dbc.Row(id='main-kpi-overview-cards', className="mb-3 justify-content-center"),
                html.Hr(),
                html.Div(id='main-scenario-recommendation-output', className="mt-3 border p-3", style={'minHeight': '150px'})
            ], md=9, style={'padding': '20px', 'height': 'calc(100vh - 90px - 1rem)', 'overflowY': 'auto'})
        ], className="g-0")
    ]), 

    # "My Hospital" Modal
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("My Hospital Settings")),
            dbc.ModalBody([
                dbc.Row([
                    dbc.Col([
                        html.H4("Hospital Configuration", className="mt-3 mb-3"),
                        dbc.Label("Hospital Type:"),
                        dbc.Select(
                            id='my-hospital-type-dropdown',
                            options=[{'label': ht, 'value': ht} for ht in hospital_type_options],
                            value=hospital_type_options[0] if hospital_type_options else None,
                            className="mb-3"
                        ),
                        dbc.Label("Total ICU Beds:"),
                        dbc.Input(id='my-hospital-icu-beds-input', type='number', value=20, min=0, step=1, className="mb-3"),
                    ], md=6),
                    dbc.Col([
                        html.H4("Operational Defaults", className="mt-3 mb-3"),
                        dbc.Label("Default ICU Occupancy (%):"),
                        dbc.Input(id='my-hospital-icu-occupancy-input', type='number', value=75, min=0, max=100, step=1, className="mb-3"),
                        dbc.Label("Default Critical Equipment Down Time (%):"),
                        dbc.Input(id='my-hospital-equip-down-input', type='number', value=2, min=0, max=100, step=1, className="mb-3"),
                        html.Br(),
                        dbc.Button("Manage Hospital Data (Future)", id='manage-hospital-data-button', disabled=True, color="secondary", className="mt-4")
                    ], md=6)
                ], className="p-3")
            ]),
        ],
        id="my-hospital-modal",
        size="lg",
        is_open=False,
    ),

    # "About" Modal
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("About ShiftSmart")),
            dbc.ModalBody([
                html.P("ShiftSmart: AI for Smarter Hospital Staffing utilizes machine learning to predict key performance indicators and recommend optimal staffing levels for emergency departments."),
                html.P("This tool is designed to help hospital administrators make data-driven decisions to improve patient flow, reduce wait times, and enhance overall operational efficiency."),
                html.P(f"App Version: 1.0.0 (Build {date.today().strftime('%Y%m%d')})")
            ]),
        ],
        id="about-modal",
        is_open=False,
    ),

], fluid=True, style={'fontFamily': 'Roboto, sans-serif'}, className='app-container')

# --- New Main Callback for Operations Dashboard Live Updates ---
@app.callback(
    Output('main-kpi-overview-cards', 'children'),
    [Input('main-patients-arrived', 'value'),
     Input('main-avg-acuity', 'value'),
     Input('main-doctors-on-shift', 'value'),
     Input('main-nurses-on-shift', 'value'),
     Input('main-day-of-week-selector', 'value'),
     Input('main-hour-of-day-slider', 'value')
    ],
    [State('my-hospital-type-dropdown', 'value'),
     State('my-hospital-icu-beds-input', 'value'),
     State('my-hospital-icu-occupancy-input', 'value'),
     State('my-hospital-equip-down-input', 'value')]
)
def update_operations_dashboard(patients_arrived, avg_acuity, doctors_on_shift, nurses_on_shift,
                                day_of_week_num, hour_of_day, 
                                hospital_type, icu_beds_total, icu_occupancy_pct, equip_down_pct_time):
    
    if isinstance(day_of_week_num, str):
        try:
            day_of_week_num = int(day_of_week_num)
        except ValueError:
            day_of_week_num = 0 
    
    # --- Default placeholder boxes for 3x2 grid ---
    default_kpi_boxes = []
    placeholder_kpi_names = [
        "Pred. Arrival to Provider (min)", "Pred. ED Length of Stay (min)", "Pred. Left Without Being Seen (%)",
        "Pred. SEP-1 Compliance (%)", "Pred. Patient Satisfaction (%)", "Pred. Readmission Rate (%)"
    ]
    for name in placeholder_kpi_names:
        box_content = html.Div([
            html.H6(name, className="text-center mb-2", style={'fontSize': '0.9em'}),
            dcc.Graph(figure=create_gauge_pie(None, name), config={'displayModeBar': False}), # Pass None to get empty figure
            html.P("N/A", className="text-center fw-bold mt-1")
        ], style={'border': '1px solid #eee', 'borderRadius': '5px', 'padding': '10px'})
        default_kpi_boxes.append(dbc.Col(box_content, md=4, className="mb-3"))

    if df.empty or not models_loaded_successfully:
        return default_kpi_boxes

    predicted_kpi_values = {}
    kpi_boxes_display = []
    try:
        scenario_dict = {
            'patients_arrived': patients_arrived, 'avg_acuity_level': avg_acuity,
            'doctors_on_shift': doctors_on_shift, 'nurses_on_shift': nurses_on_shift,
            'hospital_type': hospital_type, 'icu_beds_total': icu_beds_total,
            'icu_occupancy_pct': icu_occupancy_pct,
            'critical_equipment_down_pct_time': (equip_down_pct_time / 100.0) if equip_down_pct_time is not None else 0.0,
            'hour_of_day': hour_of_day, 'day_of_week': day_of_week_num,
            'is_weekend': (day_of_week_num in [5, 6])
        }
        for col in feature_columns:
            if col not in scenario_dict:
                 scenario_dict[col] = np.nan
        
        scenario_df = pd.DataFrame([scenario_dict])
        for cat_col in categorical_features:
             if cat_col in scenario_df.columns:
                 if cat_col in df.columns and hasattr(df[cat_col], 'cat'):
                     scenario_df[cat_col] = pd.Categorical(scenario_df[cat_col], categories=df[cat_col].cat.categories)
                 else:
                      scenario_df[cat_col] = scenario_df[cat_col].astype('category')

        for kpi_name, model in trained_models.items():
            if model is None or kpi_name not in kpi_preprocessors:
                predicted_kpi_values[kpi_name] = None
                continue
            preprocessor_kpi = kpi_preprocessors[kpi_name]
            prepared_scenario = preprocessor_kpi.transform(scenario_df[feature_columns])
            
            if kpi_configs[kpi_name]['type'] == 'classification' and hasattr(model, "predict_proba"):
                proba = model.predict_proba(prepared_scenario)
                if 1 in model.classes_:
                    class_1_idx = np.where(model.classes_ == 1)[0]
                    predicted_kpi_values[kpi_name] = proba[0, class_1_idx[0]] if class_1_idx.size > 0 else 0.0
                else:
                    predicted_kpi_values[kpi_name] = 0.0
            else:
                predicted_kpi_values[kpi_name] = model.predict(prepared_scenario)[0]

        # pred_kpi_map is now globally defined as PRED_KPI_MAP_CONFIG
        
        for kpi_internal_name, config in PRED_KPI_MAP_CONFIG.items(): # Use global config
            value = predicted_kpi_values.get(kpi_internal_name)
            val_str = "Error"
            pie_value_for_graph = None

            if value is not None:
                f_str = config["format"]
                current_value_display = value 

                if kpi_internal_name == 'readmissions_within_30_days': 
                    current_value_display *= 100
                    pie_value_for_graph = current_value_display 
                elif kpi_internal_name == 'patient_satisfaction_score': 
                    current_value_display = max(0, min((value / 50.0) * 100, 100))
                    pie_value_for_graph = current_value_display
                elif 'pct' in kpi_internal_name: 
                    pie_value_for_graph = value # Assumed to be 0-100
                    current_value_display = value # No change needed for display if already %
                else: # Time based
                     pie_value_for_graph = value
                     current_value_display = value
                
                val_str = f"{current_value_display:{f_str}}{config['unit']}"
            
            # The color passed to create_gauge_pie is the NEUTRAL color from config
            # Conditional coloring (green/red) is now handled INSIDE create_gauge_pie
            box_content = html.Div([
                html.H6(config["name"], className="text-center mb-2", style={'fontSize': '0.9em', 'fontWeight': 'bold'}),
                dcc.Graph(figure=create_gauge_pie(
                                pie_value_for_graph, 
                                config["name"], # Title (though not used in pie itself anymore)
                                config["unit"], 
                                config["max_val"], 
                                config["color"], # Neutral color
                                config["lower_is_better"],
                                config.get("target_value"),
                                config.get("icon_unicode") # Pass icon_unicode
                          ),
                          config={'displayModeBar': False}, style={'height': '120px'}),
                html.P(val_str, className="text-center fw-bold mt-1", style={'fontSize': '1.1em'})
            ], style={'border': '1px solid #ddd', 'borderRadius': '8px', 'padding': '15px', 'backgroundColor': '#fdfdfd'})
            kpi_boxes_display.append(dbc.Col(box_content, md=4, className="mb-3"))

        if not kpi_boxes_display: 
            kpi_boxes_display = default_kpi_boxes

    except Exception as e:
        print(f"Error updating operations dashboard: {e}")
        kpi_boxes_display = default_kpi_boxes
    
    return kpi_boxes_display

# --- Helper function for empty figures (retained) ---
def create_gauge_pie(value, title, unit="", max_val=100, color='#34495E', lower_is_better=False, target_value=None, icon_unicode=None):
        
        color_good = '#28A745'  
        color_bad = '#DC3545'    
        neutral_color = color    

        current_slice_color = neutral_color
        show_icon = False

        if value is not None:
            show_icon = True # Show icon if there is a value
            if target_value is not None:
                if lower_is_better:
                    if value <= target_value:
                        current_slice_color = color_good
                    elif value > target_value * 1.2: 
                        current_slice_color = color_bad
                else: # Higher is better
                    if value >= target_value:
                        current_slice_color = color_good
                    elif value < target_value * 0.8: 
                        current_slice_color = color_bad
        
        if value is None: 
            fig = go.Figure()
            fig.add_annotation(x=0.5, y=0.5, text="N/A", showarrow=False, xref='paper', yref='paper', font=dict(size=20))
            fig.update_layout(xaxis_visible=False, yaxis_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=120, margin=dict(l=10, r=10, t=10, b=10))
            return fig

        fig = go.Figure()
        capped_value = max(0, min(value, max_val))
        remainder = max(0, max_val - capped_value)
        
        pie_values = [capped_value, remainder]
        pie_colors = [current_slice_color, 'lightgrey'] 

        fig.add_trace(go.Pie(
            values=pie_values, 
            hole=.75, 
            marker_colors=pie_colors,
            textinfo='none',
            hoverinfo='skip',
            sort=False,
            direction='clockwise'
        ))
        
        annotations_list = []
        if show_icon and icon_unicode:
            annotations_list.append(
                dict(
                    text=icon_unicode,
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(family='FontAwesome', size=26, color=current_slice_color), # Using slice color for icon
                    xref="paper", yref="paper", yanchor="middle", xanchor="center"
                )
            )

        fig.update_layout(
            height=120, 
            margin=dict(l=10, r=10, t=10, b=10), 
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=annotations_list
        )
        return fig

# --- Callbacks for Stepper Controls ---

# Patients Arrived Stepper Callback
@app.callback(
    [Output('main-patients-arrived', 'value'),
     Output('display-patients-arrived', 'value')],
    [Input('plus-patients-arrived-button', 'n_clicks'),
     Input('minus-patients-arrived-button', 'n_clicks')],
    [State('main-patients-arrived', 'value')],
    prevent_initial_call=True
)
def update_patients_arrived_stepper(plus_clicks, minus_clicks, current_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_value, current_value
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    new_value = current_value if current_value is not None else 10 # Default if None

    if button_id == 'plus-patients-arrived-button':
        new_value += 1
    elif button_id == 'minus-patients-arrived-button':
        new_value -= 1
    
    if new_value < 0:
        new_value = 0
        
    return new_value, new_value

# Average Acuity Level Stepper Callback
@app.callback(
    [Output('main-avg-acuity', 'value'),
     Output('display-avg-acuity', 'value')],
    [Input('plus-avg-acuity-button', 'n_clicks'),
     Input('minus-avg-acuity-button', 'n_clicks')],
    [State('main-avg-acuity', 'value')],
    prevent_initial_call=True
)
def update_avg_acuity_stepper(plus_clicks, minus_clicks, current_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_value, current_value
        
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    new_value = current_value if current_value is not None else 3.5 # Default if None
    step = 0.1

    if button_id == 'plus-avg-acuity-button':
        new_value += step
    elif button_id == 'minus-avg-acuity-button':
        new_value -= step
    
    if new_value < 1:
        new_value = 1.0
    elif new_value > 5:
        new_value = 5.0
        
    # Ensure new_value is rounded to one decimal place due to potential float inaccuracies
    return round(new_value, 1), round(new_value, 1)

# Doctors on Shift Stepper Callback
@app.callback(
    [Output('main-doctors-on-shift', 'value'),
     Output('display-doctors-on-shift', 'value')],
    [Input('plus-doctors-on-shift-button', 'n_clicks'),
     Input('minus-doctors-on-shift-button', 'n_clicks')],
    [State('main-doctors-on-shift', 'value')],
    prevent_initial_call=True
)
def update_doctors_on_shift_stepper(plus_clicks, minus_clicks, current_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_value, current_value
        
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    new_value = current_value if current_value is not None else 3 # Default if None

    if button_id == 'plus-doctors-on-shift-button':
        new_value += 1
    elif button_id == 'minus-doctors-on-shift-button':
        new_value -= 1
    
    if new_value < 1:
        new_value = 1
        
    return new_value, new_value

# Nurses on Shift Stepper Callback
@app.callback(
    [Output('main-nurses-on-shift', 'value'),
     Output('display-nurses-on-shift', 'value')],
    [Input('plus-nurses-on-shift-button', 'n_clicks'),
     Input('minus-nurses-on-shift-button', 'n_clicks')],
    [State('main-nurses-on-shift', 'value')],
    prevent_initial_call=True
)
def update_nurses_on_shift_stepper(plus_clicks, minus_clicks, current_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_value, current_value
        
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    new_value = current_value if current_value is not None else 6 # Default if None

    if button_id == 'plus-nurses-on-shift-button':
        new_value += 1
    elif button_id == 'minus-nurses-on-shift-button':
        new_value -= 1
    
    if new_value < 1:
        new_value = 1
        
    return new_value, new_value

# --- Callbacks for Modals ---

# Callback for "My Hospital" Modal
@app.callback(
    Output("my-hospital-modal", "is_open"),
    [Input("my-hospital-button", "n_clicks")],
    [State("my-hospital-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_my_hospital_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback for "About" Modal
@app.callback(
    Output("about-modal", "is_open"),
    [Input("about-button", "n_clicks")],
    [State("about-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_about_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# --- Callback for Staffing Recommendation --- Reinstated callback
@app.callback(
    Output('main-scenario-recommendation-output', 'children'),
    Input('main-recommend-staffing-button', 'n_clicks'),
    [State('main-patients-arrived', 'value'),
     State('main-avg-acuity', 'value'),
     State('main-doctors-on-shift', 'value'),
     State('main-nurses-on-shift', 'value'),
     State('my-hospital-type-dropdown', 'value'),
     State('my-hospital-icu-beds-input', 'value'),
     State('my-hospital-icu-occupancy-input', 'value'),
     State('my-hospital-equip-down-input', 'value'),
     State('main-hour-of-day-slider', 'value'),
     State('main-day-of-week-selector', 'value')]
)
def update_recommendation(n_clicks, patients, acuity, doctors, nurses, hosp_type, 
                          icu_beds, icu_occ, equip_down_pct, hour, day_num):
    if n_clicks is None or n_clicks == 0: 
        return dbc.Alert("Click 'Get Staffing Recommendation' to see suggestions.", color="info", className="mt-3")
        
    if not models_loaded_successfully:
         return dbc.Alert("Error: Models could not be loaded. Cannot perform recommendation.", color="danger", className="mt-3")

    try:
        # Ensure day_num is int
        if isinstance(day_num, str):
            day_num = int(day_num)

        scenario_dict = {
            'patients_arrived': patients, 'avg_acuity_level': acuity, 'doctors_on_shift': doctors, 'nurses_on_shift': nurses,
            'hospital_type': hosp_type, 'icu_beds_total': icu_beds, 'icu_occupancy_pct': icu_occ,
            'critical_equipment_down_pct_time': equip_down_pct / 100.0 if equip_down_pct is not None else 0.0,
            'hour_of_day': hour, 'day_of_week': day_num, 'is_weekend': (day_num in [5, 6])
        }
        for col in feature_columns:
            if col not in scenario_dict:
                 scenario_dict[col] = None # Or np.nan, model preprocessor should handle
        input_scenario_series = pd.Series(scenario_dict)
        
        doc_adj_range = range(-2, 3)  
        nurse_adj_range = range(-3, 4) 
        min_docs_rec = 1
        min_nurses_rec = 1

        recommendation_output = recommend_optimal_staffing(
            input_scenario_features_original=input_scenario_series,
            kpi_models_dict=trained_models,
            kpi_preprocessors_dict=kpi_preprocessors,
            kpi_configs_dict=kpi_configs,
            feature_columns_list=feature_columns, 
            categorical_features_list=categorical_features, 
            doctor_adjustment_range=doc_adj_range,
            nurse_adjustment_range=nurse_adj_range,
            min_doctors=min_docs_rec,
            min_nurses=min_nurses_rec
        )
        
        rec_docs = recommendation_output['recommended_doctors']
        rec_nurses = recommendation_output['recommended_nurses']
        rec_kpis = recommendation_output['predicted_kpis_for_recommendation']
        baseline_kpis = recommendation_output['baseline_predicted_kpis']

        output_elements = [html.H5("Staffing Recommendation:", className="mb-3")]
        recommendation_alert_color = "#A3E4D7" # Teal/mint
        recommendation_alert_style = {
            'backgroundColor': recommendation_alert_color, 
            'color': '#1A5276', # Darker blue text for contrast
            'borderColor': recommendation_alert_color,
            'padding': '10px', 
            'borderRadius': '5px'
        }
        output_elements.append(dbc.Alert([
            html.Strong(f"Recommended: {rec_docs} Doctors, {rec_nurses} Nurses")
        ], style=recommendation_alert_style, className="mb-3"))

        kpi_list_items = []
        lower_is_better_kpis = ['arrival_to_provider_time_min', 'ed_length_of_stay_min', 'left_without_being_seen_pct', 'readmissions_within_30_days']
        
        # Use the same pred_kpi_map for consistent display names, units, formatting
        kpi_display_configs = PRED_KPI_MAP_CONFIG # Use global config

        for kpi_internal_name, value in rec_kpis.items():
            if value is not None and kpi_internal_name in kpi_display_configs:
                config = kpi_display_configs[kpi_internal_name]
                unit = config["unit"]
                f_str = config["format"]
                display_name = config["name"]
                current_style = {}
                baseline_value = baseline_kpis.get(kpi_internal_name)
                val_for_display = value

                # Apply scaling for display (similar to KPI cards)
                if kpi_internal_name == 'readmissions_within_30_days': val_for_display *= 100
                elif kpi_internal_name == 'patient_satisfaction_score': val_for_display = max(0, min((value / 50.0) * 100, 100))
                # 'pct' in kpi_internal_name already 0-100

                if baseline_value is not None:
                    # Note: baseline_value for probabilities/scores also needs same scaling for comparison
                    baseline_display_val = baseline_value
                    if kpi_internal_name == 'readmissions_within_30_days': baseline_display_val *= 100
                    elif kpi_internal_name == 'patient_satisfaction_score': baseline_display_val = max(0, min((baseline_value / 50.0) * 100, 100))

                    if kpi_internal_name in lower_is_better_kpis:
                        if val_for_display < baseline_display_val: current_style = {'color': '#28A745'} # Green
                        elif val_for_display > baseline_display_val: current_style = {'color': '#DC3545'} # Red
                    else: # Higher is better
                        if val_for_display > baseline_display_val: current_style = {'color': '#28A745'}
                        elif val_for_display < baseline_display_val: current_style = {'color': '#DC3545'}
                
                kpi_text = f"{display_name}: {val_for_display:{f_str}}{unit}"
                if baseline_value is not None:
                    kpi_text += f" (Baseline: {baseline_display_val:{f_str}}{unit})"

                kpi_list_items.append(dbc.ListGroupItem(kpi_text, style=current_style))
            else:
                 kpi_list_items.append(dbc.ListGroupItem(f"{kpi_internal_name.replace('_', ' ').title()}: Error or N/A"))
        
        output_elements.append(html.P("Predicted KPIs for Recommendation:", className="fw-bold mt-3"))
        output_elements.append(dbc.ListGroup(kpi_list_items, flush=True))

        return html.Div(output_elements, className="mt-2")
        
    except Exception as e:
        print(f"Error during recommendation: {e}")
        return dbc.Alert(f"An error occurred generating the recommendation: {str(e)}", color="danger", className="mt-3")

# 5. Run the Dash app
if __name__ == '__main__':
    print(f"Dash app running on http://127.0.0.1:8050/")
    app.run(debug=True) 