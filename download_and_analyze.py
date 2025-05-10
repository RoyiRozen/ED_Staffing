import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import string

# Dataset types and their descriptions
DATASETS = {
    'hospital_compare': {
        'desc': 'Hospital General Information'
    },
    'hac_reduction': {
        'desc': 'Hospital-Acquired Condition Reduction Program'
    },
    'sepsis_bundle': {
        'desc': 'Medicare Sepsis Bundle (SEP-1) Compliance Data'
    },
    'readmission_patient_experience': {
        'desc': 'Readmission Rates and Patient Experience Scores'
    }
}

DATA_DIR = Path('data')
SUMMARY_DIR = Path('summaries')
DATA_DIR.mkdir(exist_ok=True)
SUMMARY_DIR.mkdir(exist_ok=True)

def generate_random_hospital_name():
    """Generate a random hospital name"""
    prefixes = ["Memorial", "Community", "Regional", "University", "St.", "General", "Metro", 
                "County", "Providence", "Sacred Heart", "Mercy", "Holy Cross", "Valley"]
    suffixes = ["Hospital", "Medical Center", "Healthcare", "Health System", "Clinic", 
                "Memorial Hospital", "Health", "Care Center"]
    
    return f"{random.choice(prefixes)} {random.choice(suffixes)}"

def generate_random_address():
    """Generate a random address"""
    streets = ["Main St", "Oak Ave", "Maple Rd", "Washington Blvd", "Park Ave", 
               "Cedar Ln", "Elm St", "River Rd", "Lake Dr", "Highland Ave"]
    cities = ["Springfield", "Riverside", "Fairview", "Georgetown", "Salem", 
              "Franklin", "Greenville", "Bristol", "Madison", "Clinton"]
    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", 
              "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS"]
    
    return {
        "address": f"{random.randint(100, 9999)} {random.choice(streets)}",
        "city": random.choice(cities),
        "state": random.choice(states),
        "zip": f"{random.randint(10000, 99999)}"
    }

def generate_synthetic_data(dataset_type, dest, num_rows=500):
    """Generate synthetic data based on the dataset type"""
    print(f"Generating synthetic {dataset_type} data...")
    
    # Common data for all datasets
    hospital_ids = [f"H{i:04d}" for i in range(1, num_rows + 1)]
    hospital_names = [generate_random_hospital_name() for _ in range(num_rows)]
    addresses = [generate_random_address() for _ in range(num_rows)]
    
    if dataset_type == 'hospital_compare':
        # Hospital General Information
        data = {
            'provider_id': hospital_ids,
            'hospital_name': hospital_names,
            'address': [a['address'] for a in addresses],
            'city': [a['city'] for a in addresses],
            'state': [a['state'] for a in addresses],
            'zip_code': [a['zip'] for a in addresses],
            'hospital_type': np.random.choice(['Acute Care', 'Critical Access', 'Children\'s', 'Psychiatric'], num_rows),
            'hospital_ownership': np.random.choice(['Nonprofit', 'For Profit', 'Government', 'Physician'], num_rows),
            'emergency_services': np.random.choice(['Yes', 'No'], num_rows),
            'beds': np.random.randint(10, 1000, num_rows),
            'accreditation': np.random.choice(['Joint Commission', 'DNV GL', 'CIHQ', 'None'], num_rows)
        }
    
    elif dataset_type == 'hac_reduction':
        # Hospital-Acquired Condition Reduction Program
        data = {
            'provider_id': hospital_ids,
            'hospital_name': hospital_names,
            'total_hac_score': np.round(np.random.uniform(1, 10, num_rows), 2),
            'hai_score': np.round(np.random.uniform(1, 10, num_rows), 2),
            'psi_score': np.round(np.random.uniform(1, 10, num_rows), 2),
            'payment_reduction': np.random.choice(['Yes', 'No'], num_rows),
            'reduction_percentage': np.random.choice([0, 1, 2, 3], num_rows),
            'reporting_period': np.random.choice(['2021-2022', '2022-2023'], num_rows),
            'worse_than_national': np.random.choice(['Yes', 'No', 'Same'], num_rows)
        }
    
    elif dataset_type == 'sepsis_bundle':
        # Medicare Sepsis Bundle (SEP-1) Compliance Data
        data = {
            'provider_id': hospital_ids,
            'hospital_name': hospital_names,
            'compliance_rate': np.round(np.random.uniform(30, 100, num_rows), 2),
            'national_average': np.repeat(np.round(np.random.uniform(55, 65, 1), 2), num_rows),
            'cases_reviewed': np.random.randint(10, 500, num_rows),
            'compliant_cases': [],
            'measurement_period': np.random.choice(['Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022'], num_rows),
            'timely_lactate': np.round(np.random.uniform(40, 100, num_rows), 2),
            'timely_antibiotics': np.round(np.random.uniform(50, 100, num_rows), 2),
            'timely_fluids': np.round(np.random.uniform(60, 100, num_rows), 2)
        }
        
        # Calculate compliant cases based on compliance rate and cases reviewed
        for i in range(num_rows):
            compliant = int(data['cases_reviewed'][i] * data['compliance_rate'][i] / 100)
            data['compliant_cases'].append(compliant)
    
    else:  # readmission_patient_experience
        # Readmission Rates and Patient Experience Scores
        data = {
            'provider_id': hospital_ids,
            'hospital_name': hospital_names,
            'readmission_rate': np.round(np.random.uniform(5, 25, num_rows), 2),
            'national_readmission_avg': np.repeat(np.round(np.random.uniform(15, 17, 1), 2), num_rows),
            'patient_experience_score': np.round(np.random.uniform(1, 5, num_rows), 1),
            'communication_score': np.round(np.random.uniform(1, 5, num_rows), 1),
            'cleanliness_score': np.round(np.random.uniform(1, 5, num_rows), 1),
            'recommend_hospital': np.round(np.random.uniform(50, 100, num_rows), 1),
            'survey_response_rate': np.round(np.random.uniform(10, 60, num_rows), 1),
            'survey_sample_size': np.random.randint(100, 1000, num_rows)
        }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(dest, index=False)
    print(f"Generated {len(df)} rows of synthetic data to {dest}")

def analyze_csv(csv_path, summary_path, nrows=100000):
    print(f"Analyzing {csv_path}...")
    try:
        df = pd.read_csv(csv_path, nrows=nrows)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return
    
    with open(summary_path, 'w') as f:
        f.write(f"Summary for {csv_path.name}\n\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Column info:\n")
        df.info(buf=f)
        f.write("\n\nMissing values per column:\n")
        f.write(str(df.isnull().sum()))
        f.write("\n\nSample rows:\n")
        f.write(str(df.head(5)))
        f.write("\n\nBasic stats (numeric columns):\n")
        f.write(str(df.describe(include='all')))
    print(f"Summary saved to {summary_path}")

def main():
    for key, meta in DATASETS.items():
        desc = meta['desc']
        csv_path = DATA_DIR / f"{key}.csv"
        summary_path = SUMMARY_DIR / f"{key}_summary.txt"
        print(f"\n=== {desc} ===")
        generate_synthetic_data(key, csv_path)
        analyze_csv(csv_path, summary_path)

if __name__ == '__main__':
    main() 