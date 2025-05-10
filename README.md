# Hospital Performance Data Analysis

This script downloads and analyzes hospital performance data from the Centers for Medicare & Medicaid Services (CMS) data.cms.gov API.

## Datasets

The script downloads and analyzes the following datasets:

1. **Hospital Compare General Information** - Basic information about hospitals
2. **Hospital-Acquired Condition Reduction Program** - Data on hospital-acquired conditions
3. **Medicare Sepsis Bundle (SEP-1) Compliance Data** - Information on sepsis bundle compliance
4. **Readmission Rates and Patient Experience Scores** - Data on hospital readmissions and patient satisfaction

## Requirements

- Python 3.6+
- Required packages:
  - pandas
  - requests

## Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with:

```bash
python download_and_analyze.py
```

The script will:
1. Create `data/` and `summaries/` directories if they don't exist
2. Download the datasets from the CMS API
3. Save the raw data as CSV files in the `data/` directory
4. Generate summary reports in the `summaries/` directory

## Output

For each dataset, the script generates a summary file containing:
- Dataset shape (rows and columns)
- Column information
- Missing value counts
- Sample rows
- Basic statistics for numeric columns

## Notes

- The script uses the CMS Data.gov API to download the datasets
- A limit of 100,000 rows is applied to each dataset to manage memory usage
- The script includes error handling for API requests and data processing
- A small delay is added between requests to avoid rate limiting 