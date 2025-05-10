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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "ED Operations & Staffing Optimization Dashboard"

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
else:
    print("DataFrame is empty. Filters will have default/empty options.")

# --- KPI list for Distribution Analysis Selector ---
# Define which columns are suitable for distribution analysis
distribution_kpi_options = [
    'arrival_to_provider_time_min', 'ed_length_of_stay_min', 
    'left_without_being_seen_pct', 'sep1_compliance_pct', 
    'patient_satisfaction_score', 'icu_occupancy_pct', 
    'diagnostic_imaging_avg_wait_time_min', 'critical_equipment_down_pct_time',
    'patients_arrived', 'avg_acuity_level', 'doctors_on_shift', 'nurses_on_shift'
] if not df.empty else [] # Ensure options are empty if df failed to load

# --- Helper function for creating KPI cards ---
def create_kpi_card(title, value_id, card_style=None):
    default_style = {
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'padding': '15px',
        'textAlign': 'center',
        'margin': '5px',
        'minWidth': '150px', # Ensure cards have a minimum width
        'flex': '1' # Allow cards to grow and shrink
    }
    if card_style:
        default_style.update(card_style)
        
    return html.Div([
        html.H5(title, style={'margin': '0 0 10px 0', 'fontSize': '0.9em'}),
        html.H4(id=value_id, style={'margin': '0', 'fontSize': '1.5em'})
    ], style=default_style)

# --- Helper function for empty figures ---
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

# 3. App Layout with Tabs
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("ED Operations & Staffing Optimization Dashboard"), width=12), className="mb-4 mt-4 text-center"),
    dbc.Tabs([
        # -- Tab 1: Data Explorer --
        dbc.Tab(label='Data Explorer', children=[
            dbc.Row([
                # Sidebar
                dbc.Col([
                    html.H4("Filters & Controls", className="mt-3"), html.Hr(),
                    dbc.Label("Hospital ID(s):"),
                    dcc.Dropdown(id='hospital-id-dropdown', options=[{'label': str(i), 'value': i} for i in hospital_id_options], value=hospital_id_options if hospital_id_options else [], multi=True), html.Br(),
                    dbc.Label("Hospital Type(s):"),
                    dcc.Dropdown(id='hospital-type-dropdown', options=[{'label': str(i), 'value': i} for i in hospital_type_options], value=hospital_type_options if hospital_type_options else [], multi=True), html.Br(),
                    dbc.Label("Date Range:"),
                    dcc.DatePickerRange(id='date-picker-range', min_date_allowed=min_date_allowed, max_date_allowed=max_date_allowed, start_date=initial_start_date, end_date=initial_end_date, display_format='YYYY-MM-DD', className="d-block"), html.Br(),
                    dbc.Label("Day of Week:"),
                    dcc.Dropdown(id='day-of-week-dropdown', options=[{'label': str(i), 'value': i} for i in day_of_week_options], value=day_of_week_options if day_of_week_options else [], multi=True), html.Br(),
                    dbc.Label("Hour of Day Range:"),
                    dcc.RangeSlider(id='hour-of-day-slider', min=0, max=23, step=1, marks={i: str(i) for i in range(0, 24, 3)}, value=[0, 23], tooltip={"placement": "bottom", "always_visible": False}),
                ], md=3, style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'height': '85vh', 'overflowY': 'auto'}),
                
                # Main Content Area for Explorer Tab
                dbc.Col([
                    html.Div(id='filtered-rows-count-output'), html.Hr(),
                    html.H5("KPI Overview", className="mt-3"),
                    dbc.Row(id='kpi-overview-cards', className="mb-3"), # Cards will be added here by callback
                    html.Hr(),
                    html.H5("Time Series Analysis", className="mt-3"),
                    html.P("Showing daily averages based on selected filters.", className="text-muted small"),
                    dbc.Row([dbc.Col(dcc.Graph(id='ts-arrivals-staffing'), md=12)]),
                    dbc.Row([dbc.Col(dcc.Graph(id='ts-wait-times'), md=12)]),
                    dbc.Row([dbc.Col(dcc.Graph(id='ts-lwbs-satisfaction'), md=12)]),
                    html.Hr(),
                    html.H5("Distribution Analysis", className="mt-3"),
                    dbc.Row([
                        dbc.Col([dbc.Label("Select KPI for Distribution:"), dcc.Dropdown(id='dist-kpi-selector', options=[{'label': kpi, 'value': kpi} for kpi in distribution_kpi_options], value='ed_length_of_stay_min')], width=6),
                        dbc.Col([dbc.Label("Group Box Plot by:"), dcc.Dropdown(id='dist-grouping-selector', options=[{'label': 'Hospital Type', 'value': 'hospital_type'},{'label': 'Day of Week', 'value': 'day_of_week_name'}], value='hospital_type')], width=6)
                    ], className="mb-3"),
                    dbc.Row([dbc.Col(dcc.Graph(id='dist-histogram'), md=12)]),
                    dbc.Row([dbc.Col(dcc.Graph(id='dist-boxplot'), md=12)]),
                    dbc.Row([dbc.Col(dcc.Graph(id='dist-scatter'), md=12)]),
                ], md=9)
            ], className="mt-4")
        ]),

        # -- Tab 2: Scenario Simulator --
        dbc.Tab(label='Scenario Simulator', children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Define Scenario Inputs", className="mt-3"),
                    html.P("Enter the details for a specific hour to predict outcomes and get staffing recommendations.", className="text-muted small"),
                    dbc.Row([
                        dbc.Col([dbc.Label("Patients Arrived:"), dbc.Input(id='sim-patients-arrived', type='number', value=10, step=1, min=0)], md=6),
                        dbc.Col([dbc.Label("Avg Acuity Level (1-5):"), dbc.Input(id='sim-avg-acuity', type='number', value=3.5, step=0.1, min=1, max=5)], md=6),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([dbc.Label("Current Doctors on Shift:"), dbc.Input(id='sim-doctors-on-shift', type='number', value=3, step=1, min=1)], md=6),
                        dbc.Col([dbc.Label("Current Nurses on Shift:"), dbc.Input(id='sim-nurses-on-shift', type='number', value=6, step=1, min=1)], md=6),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([dbc.Label("Hospital Type:"), dcc.Dropdown(id='sim-hospital-type', options=[{'label': ht, 'value': ht} for ht in hospital_type_options], value=hospital_type_options[0] if hospital_type_options else None)], md=6),
                        dbc.Col([dbc.Label("ICU Beds Total:"), dbc.Input(id='sim-icu-beds', type='number', value=15, step=1, min=0)], md=6),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([dbc.Label("ICU Occupancy (%):"), dbc.Input(id='sim-icu-occupancy', type='number', value=70, min=0, max=100, step=1)], md=6),
                        dbc.Col([dbc.Label("Critical Equip Down Time (%):"), dbc.Input(id='sim-equip-down', type='number', value=0, min=0, max=100, step=1)], md=6),
                    ], className="mb-3"),
                    dbc.Row([
                         dbc.Col([dbc.Label("Day of Week:"), dcc.Dropdown(id='sim-day-of-week', options=[{'label': name, 'value': num} for num, name in day_map.items()], value=0)], md=6),
                         dbc.Col([dbc.Label("Hour of Day (0-23):"), dcc.Slider(id='sim-hour-of-day', min=0, max=23, step=1, marks={i: str(i) for i in range(0, 24, 3)}, value=10, tooltip={"placement": "bottom", "always_visible": True})], md=6),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(dbc.Button('Predict KPIs for Scenario', id='predict-kpi-button', n_clicks=0, color="primary", className="me-2"), width="auto"),
                        dbc.Col(dbc.Button('Get Staffing Recommendation', id='recommend-staffing-button', n_clicks=0, color="success"), width="auto"),
                    ], className="mb-3"),
                ], md=6), # Input section takes half width
                
                dbc.Col([
                    html.H4("Simulation Results", className="mt-3"), html.Hr(),
                    html.Div(id='scenario-prediction-output', children="Predictions will appear here."), html.Hr(),
                    html.Div(id='scenario-recommendation-output', children="Recommendations will appear here.")
                ], md=6) # Output section takes other half
            ], className="mt-4")
        ])
    ])
], fluid=True) # Use fluid container for full width

# 4. Callback to update all outputs based on filters
@app.callback(
    [Output('filtered-rows-count-output', 'children'),
     Output('kpi-overview-cards', 'children'), # Output the entire row of cards
     Output('ts-arrivals-staffing', 'figure'),
     Output('ts-wait-times', 'figure'),
     Output('ts-lwbs-satisfaction', 'figure'),
     Output('dist-histogram', 'figure'),
     Output('dist-boxplot', 'figure'),
     Output('dist-scatter', 'figure')
    ],
    [Input('hospital-id-dropdown', 'value'),
     Input('hospital-type-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('day-of-week-dropdown', 'value'),
     Input('hour-of-day-slider', 'value'),
     Input('dist-kpi-selector', 'value'),
     Input('dist-grouping-selector', 'value')]
)
def update_data_explorer_output(selected_hospital_ids, selected_hospital_types, 
                            start_date_str, end_date_str, 
                            selected_days_of_week_names, selected_hour_range,
                            selected_dist_kpi, selected_dist_grouping):
    
    # Default empty outputs
    kpi_placeholder = "N/A"
    empty_fig = create_empty_figure()
    # Create placeholder cards
    kpi_cards = [
        dbc.Col(dbc.Card([dbc.CardHeader("Avg Arrival to Provider (min)"), dbc.CardBody(kpi_placeholder)]), width=4, lg=2, className="mb-2"),
        dbc.Col(dbc.Card([dbc.CardHeader("Avg ED LOS (min)"), dbc.CardBody(kpi_placeholder)]), width=4, lg=2, className="mb-2"),
        dbc.Col(dbc.Card([dbc.CardHeader("Avg LWBS (%)"), dbc.CardBody(kpi_placeholder)]), width=4, lg=2, className="mb-2"),
        dbc.Col(dbc.Card([dbc.CardHeader("Avg SEP-1 Compliance (%)"), dbc.CardBody(kpi_placeholder)]), width=4, lg=2, className="mb-2"),
        dbc.Col(dbc.Card([dbc.CardHeader("Avg Patient Satisfaction"), dbc.CardBody(kpi_placeholder)]), width=4, lg=2, className="mb-2"),
        dbc.Col(dbc.Card([dbc.CardHeader("Readmission Rate (%)"), dbc.CardBody(kpi_placeholder)]), width=4, lg=2, className="mb-2"),
    ]
    initial_outputs = (
        html.H5("Filtered Rows: 0"), kpi_cards,
        empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
    )

    if df.empty:
        return initial_outputs

    # Filtering Logic
    filtered_df = df.copy()
    if selected_hospital_ids: filtered_df = filtered_df[filtered_df['hospital_id'].isin(selected_hospital_ids)]
    if selected_hospital_types: filtered_df = filtered_df[filtered_df['hospital_type'].isin(selected_hospital_types)]
    if start_date_str and end_date_str:
        try:
            start_date_obj = date.fromisoformat(start_date_str)
            end_date_obj = date.fromisoformat(end_date_str)
            filtered_df = filtered_df[(filtered_df['date'] >= start_date_obj) & (filtered_df['date'] <= end_date_obj)]
        except ValueError:
            error_message = "Invalid date format received."
            return (
                html.H5("Filtered Rows: 0"), kpi_cards, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
            )
    if selected_days_of_week_names: filtered_df = filtered_df[filtered_df['day_of_week_name'].isin(selected_days_of_week_names)]
    if selected_hour_range: filtered_df = filtered_df[(filtered_df['hour_of_day'] >= selected_hour_range[0]) & (filtered_df['hour_of_day'] <= selected_hour_range[1])]

    filtered_rows_count_html = html.H5(f"Filtered Rows: {len(filtered_df)}")

    # Calculate KPIs and Generate Charts if data available
    if not filtered_df.empty:
        # Calculate KPIs
        avg_atp = filtered_df['arrival_to_provider_time_min'].mean()
        avg_los = filtered_df['ed_length_of_stay_min'].mean()
        avg_lwbs = filtered_df['left_without_being_seen_pct'].mean()
        avg_sep1 = filtered_df['sep1_compliance_pct'].mean()
        avg_pss = filtered_df['patient_satisfaction_score'].mean()
        avg_readmit = filtered_df['readmissions_within_30_days'].mean() * 100
        kpi_atp_text = f"{avg_atp:.1f}"
        kpi_los_text = f"{avg_los:.1f}"
        kpi_lwbs_text = f"{avg_lwbs:.1f}%"
        kpi_sep1_text = f"{avg_sep1:.1f}%"
        kpi_pss_text = f"{avg_pss:.1f}"
        kpi_readmit_text = f"{avg_readmit:.2f}%"
        
        # Update KPI Cards
        kpi_cards = [
            dbc.Col(dbc.Card([dbc.CardHeader("Avg Arrival to Provider (min)"), dbc.CardBody(kpi_atp_text)]), width=6, lg=2, className="mb-2"),
            dbc.Col(dbc.Card([dbc.CardHeader("Avg ED LOS (min)"), dbc.CardBody(kpi_los_text)]), width=6, lg=2, className="mb-2"),
            dbc.Col(dbc.Card([dbc.CardHeader("Avg LWBS (%)"), dbc.CardBody(kpi_lwbs_text)]), width=6, lg=2, className="mb-2"),
            dbc.Col(dbc.Card([dbc.CardHeader("Avg SEP-1 Compliance (%)"), dbc.CardBody(kpi_sep1_text)]), width=6, lg=2, className="mb-2"),
            dbc.Col(dbc.Card([dbc.CardHeader("Avg Patient Satisfaction"), dbc.CardBody(kpi_pss_text)]), width=6, lg=2, className="mb-2"),
            dbc.Col(dbc.Card([dbc.CardHeader("Readmission Rate (%)"), dbc.CardBody(kpi_readmit_text)]), width=6, lg=2, className="mb-2"),
        ]
        
        # Resample for time series charts (Daily Average)
        df_resampled = filtered_df.set_index('datetime').resample('D').mean(numeric_only=True)
        
        # Create Time Series Figures
        if not df_resampled.empty:
            # Chart 1: Arrivals & Staffing
            fig_arrivals_staffing = make_subplots(specs=[[{"secondary_y": True}]])
            fig_arrivals_staffing.add_trace(go.Scatter(x=df_resampled.index, y=df_resampled['patients_arrived'], name='Avg Patients Arrived', line=dict(color='blue')), secondary_y=False)
            fig_arrivals_staffing.add_trace(go.Scatter(x=df_resampled.index, y=df_resampled['doctors_on_shift'], name='Avg Doctors', line=dict(color='red', dash='dot')), secondary_y=True)
            fig_arrivals_staffing.add_trace(go.Scatter(x=df_resampled.index, y=df_resampled['nurses_on_shift'], name='Avg Nurses', line=dict(color='green', dash='dot')), secondary_y=True)
            fig_arrivals_staffing.update_layout(title_text="Patient Arrivals & Staffing (Daily Avg)", margin=dict(l=20, r=20, t=40, b=20))
            fig_arrivals_staffing.update_yaxes(title_text="Avg Patients Arrived", secondary_y=False)
            fig_arrivals_staffing.update_yaxes(title_text="Avg Staff on Shift", secondary_y=True)

            # Chart 2: Wait Times & LOS
            fig_wait_times = px.line(df_resampled, y=['arrival_to_provider_time_min', 'ed_length_of_stay_min'], title="Wait Times & ED LOS (Daily Avg)")
            fig_wait_times.update_layout(margin=dict(l=20, r=20, t=40, b=20), legend_title_text='Metric')
            fig_wait_times.update_yaxes(title_text="Time (minutes)")

            # Chart 3: LWBS & Satisfaction
            fig_lwbs_satisfaction = make_subplots(specs=[[{"secondary_y": True}]])
            fig_lwbs_satisfaction.add_trace(go.Scatter(x=df_resampled.index, y=df_resampled['left_without_being_seen_pct'], name='Avg LWBS %', line=dict(color='orange')), secondary_y=False)
            fig_lwbs_satisfaction.add_trace(go.Scatter(x=df_resampled.index, y=df_resampled['patient_satisfaction_score'], name='Avg Patient Satisfaction', line=dict(color='purple', dash='dash')), secondary_y=True)
            fig_lwbs_satisfaction.update_layout(title_text="LWBS & Patient Satisfaction (Daily Avg)", margin=dict(l=20, r=20, t=40, b=20))
            fig_lwbs_satisfaction.update_yaxes(title_text="LWBS (%)", secondary_y=False)
            fig_lwbs_satisfaction.update_yaxes(title_text="Avg Satisfaction Score", secondary_y=True)
        else:
            fig_arrivals_staffing = fig_wait_times = fig_lwbs_satisfaction = create_empty_figure("No data for time series chart after resampling.")

        # Generate Distribution Figures
        # Histogram
        fig_dist_histogram = px.histogram(filtered_df, x=selected_dist_kpi, title=f"Distribution of {selected_dist_kpi}")
        fig_dist_histogram.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
        # Box Plot
        fig_dist_boxplot = px.box(filtered_df, x=selected_dist_grouping, y=selected_dist_kpi, color=selected_dist_grouping, title=f"{selected_dist_kpi} by {selected_dist_grouping}")
        fig_dist_boxplot.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
        # Scatter Plot
        try:
            # Calculate load factor safely
            denominator = (filtered_df['doctors_on_shift'] + filtered_df['nurses_on_shift'] * 0.5) + 1e-6
            filtered_df['load_factor'] = (filtered_df['patients_arrived'] * filtered_df['avg_acuity_level']) / denominator
            fig_dist_scatter = px.scatter(filtered_df, x='load_factor', y='left_without_being_seen_pct', color='hospital_type', title="Load Factor vs LWBS (%)")
            fig_dist_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350)
        except Exception as e:
            print(f"Error creating scatter plot: {e}")
            fig_dist_scatter = create_empty_figure("Error generating scatter plot.")

    return (
        filtered_rows_count_html, kpi_cards,
        fig_arrivals_staffing, fig_wait_times, fig_lwbs_satisfaction,
        fig_dist_histogram, fig_dist_boxplot, fig_dist_scatter
    )

# --- Callback for Scenario Simulator Tab ---
@app.callback(
    Output('scenario-prediction-output', 'children'),
    Input('predict-kpi-button', 'n_clicks'),
    [State('sim-patients-arrived', 'value'),
     State('sim-avg-acuity', 'value'),
     State('sim-doctors-on-shift', 'value'),
     State('sim-nurses-on-shift', 'value'),
     State('sim-hospital-type', 'value'),
     State('sim-icu-beds', 'value'),
     State('sim-icu-occupancy', 'value'),
     State('sim-equip-down', 'value'), # This is % input
     State('sim-hour-of-day', 'value'),
     State('sim-day-of-week', 'value')] # This is 0-6
)
def update_scenario_predictions(n_clicks, patients, acuity, doctors, nurses, hosp_type, 
                                icu_beds, icu_occ, equip_down_pct, hour, day_num):
    if n_clicks == 0:
        return dbc.Alert("Enter scenario details and click 'Predict KPIs' to see predicted outcomes.", color="info")
        
    if not models_loaded_successfully:
         return dbc.Alert("Error: Models could not be loaded. Cannot perform prediction.", color="danger")

    # Prepare input features for the models
    try:
        # Create a dictionary matching the feature_columns order/names
        scenario_dict = {
            'patients_arrived': patients,
            'avg_acuity_level': acuity,
            'doctors_on_shift': doctors,
            'nurses_on_shift': nurses,
            'hospital_type': hosp_type, 
            'icu_beds_total': icu_beds,
            'icu_occupancy_pct': icu_occ,
            # Convert equip down % to proportion (0-1)
            'critical_equipment_down_pct_time': equip_down_pct / 100.0 if equip_down_pct is not None else 0.0,
            'hour_of_day': hour,
            'day_of_week': day_num, # Use the numeric day of week (0-6)
            'is_weekend': (day_num in [5, 6]) # Derive is_weekend
        }
        
        # Ensure all expected columns are present, even if None
        for col in feature_columns:
            if col not in scenario_dict:
                 scenario_dict[col] = None # Or some default if appropriate
                 print(f"Warning: Feature '{col}' missing from scenario dict, setting to None.")
                 
        # Convert dict to DataFrame (1 row)
        scenario_df = pd.DataFrame([scenario_dict])
        
        # Set categorical dtypes BEFORE passing to preprocessor
        for cat_col in categorical_features:
             if cat_col in scenario_df.columns:
                 # Use categories from the original dataframe if possible to handle unseen values
                 if cat_col in df.columns and hasattr(df[cat_col], 'cat'):
                     scenario_df[cat_col] = pd.Categorical(scenario_df[cat_col], categories=df[cat_col].cat.categories)
                 else:
                      scenario_df[cat_col] = scenario_df[cat_col].astype('category')
        
        # Predict using each model
        predicted_kpis = {}
        for kpi_name, model in trained_models.items():
            preprocessor_kpi = kpi_preprocessors[kpi_name]
            prepared_scenario = preprocessor_kpi.transform(scenario_df[feature_columns]) # Ensure correct feature order
            
            if kpi_configs[kpi_name]['type'] == 'classification' and hasattr(model, "predict_proba"):
                proba = model.predict_proba(prepared_scenario) # proba is for one sample: shape (1, n_classes_model_knows)
                if 1 in model.classes_: # Assuming positive class is '1'
                    class_1_idx = np.where(model.classes_ == 1)[0]
                    if class_1_idx.size > 0:
                        predicted_kpis[kpi_name] = proba[0, class_1_idx[0]]
                    else:
                        print(f"Warning: Class 1 reported in model.classes_ but not found by np.where for {kpi_name} in update_scenario_predictions. Setting P(1) to 0.0.")
                        predicted_kpis[kpi_name] = 0.0
                else:
                    # Class '1' was not seen by the model during training or model.classes_ is unexpected.
                    # So, probability of observing class '1' is 0.
                    predicted_kpis[kpi_name] = 0.0
            else:
                predicted_kpis[kpi_name] = model.predict(prepared_scenario)[0]
        
        # Format output using dbc list group
        output_items = [html.H5("Predicted KPI Outcomes:")]
        list_group_items = []
        for kpi, value in predicted_kpis.items():
            unit = ""
            f_str = ".1f" # Default formatting string
            if kpi == 'readmissions_within_30_days':
                unit = " (Probability)"
                f_str = ".3f" # Probabilities often shown with more precision
            elif 'pct' in kpi:
                unit = "%"
                # f_str remains ".1f"
            elif 'min' in kpi:
                unit = " min"
                # f_str remains ".1f"
            elif 'score' in kpi:
                # f_str remains ".1f"
                pass # No change to unit or f_str needed from default for score
            list_group_items.append(dbc.ListGroupItem(f"{kpi.replace('_', ' ').title()}: {value:{f_str}}{unit}"))
        output_items.append(dbc.ListGroup(list_group_items))
        return html.Div(output_items)

    except Exception as e:
        print(f"Error during prediction: {e}") # Log error server-side
        return dbc.Alert(f"An error occurred during prediction: {e}", color="danger")

# --- Callback for Staffing Recommendation ---
@app.callback(
    Output('scenario-recommendation-output', 'children'),
    Input('recommend-staffing-button', 'n_clicks'),
    [State('sim-patients-arrived', 'value'), State('sim-avg-acuity', 'value'),
     State('sim-doctors-on-shift', 'value'), State('sim-nurses-on-shift', 'value'),
     State('sim-hospital-type', 'value'), State('sim-icu-beds', 'value'),
     State('sim-icu-occupancy', 'value'), State('sim-equip-down', 'value'),
     State('sim-hour-of-day', 'value'), State('sim-day-of-week', 'value')]
)
def update_recommendation(n_clicks, patients, acuity, doctors, nurses, hosp_type, 
                          icu_beds, icu_occ, equip_down_pct, hour, day_num):
    if n_clicks == 0:
        return dbc.Alert("Click 'Get Recommendation' to see optimal staffing suggestions.", color="info")
        
    if not models_loaded_successfully:
         return dbc.Alert("Error: Models could not be loaded. Cannot perform recommendation.", color="danger")

    # Prepare input features as a pandas Series
    try:
        scenario_dict = {
            'patients_arrived': patients, 'avg_acuity_level': acuity, 'doctors_on_shift': doctors, 'nurses_on_shift': nurses,
            'hospital_type': hosp_type, 'icu_beds_total': icu_beds, 'icu_occupancy_pct': icu_occ,
            'critical_equipment_down_pct_time': equip_down_pct / 100.0 if equip_down_pct is not None else 0.0,
            'hour_of_day': hour, 'day_of_week': day_num, 'is_weekend': (day_num in [5, 6])
        }
        # Ensure all expected columns are present
        for col in feature_columns:
            if col not in scenario_dict:
                 scenario_dict[col] = None 
        input_scenario_series = pd.Series(scenario_dict)
        
        # Define adjustment ranges and minimums for the recommendation function
        doc_adj_range = range(-2, 3)  # Allows -2, -1, 0, +1, +2 changes
        nurse_adj_range = range(-3, 4) # Allows -3, -2, -1, 0, +1, +2, +3 changes
        min_docs_rec = 1
        min_nurses_rec = 1

        # Call the recommendation function
        recommendation_output = recommend_optimal_staffing(
            input_scenario_features_original=input_scenario_series,
            kpi_models_dict=trained_models,
            kpi_preprocessors_dict=kpi_preprocessors,
            kpi_configs_dict=kpi_configs,
            feature_columns_list=feature_columns, # Pass feature list
            categorical_features_list=categorical_features, # Pass categorical list
            doctor_adjustment_range=doc_adj_range,
            nurse_adjustment_range=nurse_adj_range,
            min_doctors=min_docs_rec,
            min_nurses=min_nurses_rec
        )
        
        # Format the output using dbc components
        output_elements = [html.H5("Staffing Recommendation:")]
        
        # Baseline info
        baseline_alert_items = [html.Strong("Baseline (Current Scenario): "), f"{doctors} Doctors, {nurses} Nurses"]
        baseline_kpis = recommendation_output['baseline_predicted_kpis']
        kpi_list_baseline = []
        for kpi, value in baseline_kpis.items():
             if value is not None:
                 unit = ""; f_str = ".1f" # Default formatting string
                 if kpi == 'readmissions_within_30_days':
                     unit = " (Prob)"
                     f_str = ".3f"
                 elif 'pct' in kpi:
                     unit = "%"
                     # f_str remains ".1f"
                 elif 'min' in kpi:
                     unit = " min"
                     # f_str remains ".1f"
                 elif 'score' in kpi:
                     # f_str remains ".1f"
                     pass # No change to unit or f_str needed from default for score
                 kpi_list_baseline.append(dbc.ListGroupItem(f"{kpi.replace('_', ' ').title()}: {value:{f_str}}{unit}"))
             else:
                 kpi_list_baseline.append(dbc.ListGroupItem(f"{kpi.replace('_', ' ').title()}: Error"))
        output_elements.append(dbc.Alert(baseline_alert_items, color="secondary"))
        output_elements.append(dbc.ListGroup(kpi_list_baseline, flush=True))
        output_elements.append(html.Br())

        # Recommendation info
        rec_docs = recommendation_output['recommended_doctors']
        rec_nurses = recommendation_output['recommended_nurses']
        rec_alert_items = [html.Strong("Recommendation: "), f"{rec_docs} Doctors, {rec_nurses} Nurses"]
        rec_kpis = recommendation_output['predicted_kpis_for_recommendation']
        kpi_list_rec = []
        for kpi, value in rec_kpis.items():
            if value is not None:
                unit = ""; f_str = ".1f" # Default formatting string
                if kpi == 'readmissions_within_30_days':
                    unit = " (Prob)"
                    f_str = ".3f"
                elif 'pct' in kpi:
                    unit = "%"
                    # f_str remains ".1f"
                elif 'min' in kpi:
                    unit = " min"
                    # f_str remains ".1f"
                elif 'score' in kpi:
                    # f_str remains ".1f"
                    pass # No change to unit or f_str needed from default for score
                # Highlight the primary optimization KPI
                is_primary_kpi = (kpi == 'left_without_being_seen_pct')
                kpi_list_rec.append(dbc.ListGroupItem(f"{kpi.replace('_', ' ').title()}: {value:{f_str}}{unit}",
                                                      className="fw-bold" if is_primary_kpi else ""))
            else:
                 kpi_list_rec.append(dbc.ListGroupItem(f"{kpi.replace('_', ' ').title()}: Error"))
        output_elements.append(dbc.Alert(rec_alert_items, color="success"))
        output_elements.append(dbc.ListGroup(kpi_list_rec, flush=True))

        return html.Div(output_elements)
        
    except Exception as e:
        print(f"Error during recommendation: {e}") # Log error server-side
        return dbc.Alert(f"An error occurred during recommendation: {e}", color="danger")

# 5. Run the Dash app
if __name__ == '__main__':
    print(f"Dash app running on http://127.0.0.1:8050/")
    app.run(debug=True) 