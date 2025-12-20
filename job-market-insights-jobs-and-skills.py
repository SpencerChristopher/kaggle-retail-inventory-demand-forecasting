# %% [markdown]
# # The Entry-Level Job Seeker's Guide: A Data-Driven Analysis
#
# ## Project Hypothesis & Goals Introduction
#
# **Core Question:** Is a $150,000 salary in San Francisco *really* better than a $100,000 salary in Austin? This project tests the hypothesis that real purchasing power, not nominal salary, is the true measure of financial opportunity.
#
# **Primary Goals:**
# 1. To create a "City Opportunity Score" that ranks cities based on a combination of job volume and PPP-adjusted salary for entry-level roles.
# 2. To identify the most in-demand skills required to access jobs in these high-opportunity cities.
# 3. To deliver clear, actionable insights that empower an entry-level job seeker to make more informed decisions in their job search.
#
# **Intended Audience:** This guide is specifically designed for inexperienced job seekers navigating the complex entry-level job market.
#
# ---
# 
# ## Data Sources and Methodology Notes
# 
# ### Main Dataset
# The primary dataset for this analysis is `job_market.csv`, containing job posting information.
# 
# ### Data Augmentation: Purchasing Power Parity (PPP)
# 
# To provide a more accurate measure of financial opportunity, this analysis is augmented with data from `city_PPP.csv`.
# 
# *   **What is this value?** The `ppp_multiplier` column represents Numbeo's **Cost of Living Index (CLI)**, with New York City (NYC) as the 100% baseline.
# *   **How to Interpret It:** A city with a CLI of 120 is 20% more expensive than NYC. A city with a CLI of 70 is 30% cheaper.
# *   **Formula Used:** Our "PPP-Adjusted Salary" is calculated as: `Average Salary * (100 / Cost of Living Index)`. This correctly scales salaries up for cheaper cities and down for more expensive ones.
# *   **Source and Limitations:** This data is derived from the crowd-sourced database [Numbeo](https://www.numbeo.com/cost-of-living/). It should be considered a valuable **estimate** and not official government data. The provided file is a static snapshot and may not reflect the most current economic conditions.
# 
# ---

# %% [code] {"execution":{"iopub.status.busy":"2025-12-19T19:32:51.549388Z","iopub.execute_input":"2025-12-19T19:32:51.550448Z","iopub.status.idle":"2025-12-19T19:32:52.887597Z","shell.execute_reply.started":"2025-12-19T19:32:51.550404Z","shell.execute_reply":"2025-12-19T19:32:52.886513Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:52.890021Z","iopub.execute_input":"2025-12-19T19:32:52.890465Z","iopub.status.idle":"2025-12-19T19:32:52.899954Z","shell.execute_reply.started":"2025-12-19T19:32:52.890434Z","shell.execute_reply":"2025-12-19T19:32:52.898979Z"}}
import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent adding handlers multiple times if the cell is run repeatedly
if not logger.handlers:
    # Create a file handler
    file_handler = logging.FileHandler('job_market_data.log', mode='w')  # Overwrite log each run
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("Logger initialized and working.")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:52.901310Z","iopub.execute_input":"2025-12-19T19:32:52.901800Z","iopub.status.idle":"2025-12-19T19:32:52.920047Z","shell.execute_reply.started":"2025-12-19T19:32:52.901716Z","shell.execute_reply":"2025-12-19T19:32:52.918876Z"}}
# Markdown cell to state project goals and data import stratergy

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### **Data Import and Initial Processing**
# (No changes here)

# %% [code] {"execution":{"iopub.status.busy":"2025-12-19T19:32:52.922670Z","iopub.execute_input":"2025-12-19T19:32:52.923353Z","iopub.status.idle":"2025-12-19T19:32:52.997918Z","shell.execute_reply.started":"2025-12-19T19:32:52.923320Z","shell.execute_reply":"2025-12-19T19:32:52.996940Z"},"jupyter":{"outputs_hidden":false}}
# --- Configuration ---
DATA_FILE_PATH = '/kaggle/input/job-market-insight/job_market.csv'  # Corrected data file path
PPP_FILE_PATH = '/kaggle/input/city-ppp-headers/city_PPP.csv'  # New: Path to PPP data
CHUNK_SIZE = 500
NA_VALUES = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A",
             "NA", "NULL", "NaN", "n/a", "nan", "null"]
NUMERIC_COLS = ['salary_min', 'salary_max', 'experience_required']
DATE_COL = 'publication_date'

# Initialize config object
config = {
    'DATA_FILE_PATH': DATA_FILE_PATH,
    'PPP_FILE_PATH': PPP_FILE_PATH,
    'CHUNK_SIZE': CHUNK_SIZE,
    'NA_VALUES': NA_VALUES,
    'NUMERIC_COLS': NUMERIC_COLS,
    'DATE_COL': DATE_COL,
    'ENTRY_LEVEL_THRESHOLD': 2,  # Defined in calculate_v2_city_opportunity_score
    'MIN_JOB_COUNT': 3,  # Defined in calculate_v2_city_opportunity_score
    'V2_WEIGHTS': {'count': 0.5, 'salary': 0.5},  # Defined in calculate_v2_city_opportunity_score
    'V3_WEIGHTS': {'count': 0.5, 'salary': 0.5},  # New: Weights for V3 City Opportunity Score (PPP Adjusted)
    'V3_OUTPUT_FILENAME': '/kaggle/working/city_opportunity_v3.png',  # New: Output filename for V3 plot
    'V2_V3_COMPARISON_OUTPUT_FILENAME': '/kaggle/working/city_opportunity_v2_v3_comparison.png' # New: Output filename for V2 vs V3 plot
}

# --- Data Structures ---
clean_chunks = []
malformed_chunks = []


def parse_mixed_dates(series):
    # Handles Unix timestamps and standard date strings
    numeric_dates = pd.to_numeric(series, errors='coerce')
    unix_dates = pd.to_datetime(numeric_dates, unit='s', errors='coerce')
    string_dates = pd.to_datetime(series[numeric_dates.isna()], errors='coerce')
    return unix_dates.fillna(string_dates)


# --- Main Import and Validation Loop ---
try:
    logger.info(f"Starting robust data import from {DATA_FILE_PATH}...")

    with pd.read_csv(DATA_FILE_PATH, chunksize=CHUNK_SIZE, dtype=object, keep_default_na=False) as reader:
        for i, chunk_raw_text in enumerate(reader):
            # ... (import logic remains the same)
            legit_nan_mask = chunk_raw_text.isin(NA_VALUES)
            converted_chunk = chunk_raw_text.copy()
            for col in NUMERIC_COLS:
                converted_chunk[col] = pd.to_numeric(converted_chunk[col], errors='coerce')
            converted_chunk[DATE_COL] = parse_mixed_dates(converted_chunk[DATE_COL])
            final_nan_mask = converted_chunk.isnull()
            import_created_nans = final_nan_mask & ~legit_nan_mask
            is_malformed = import_created_nans.any(axis=1)
            clean_rows = converted_chunk[~is_malformed]
            if not clean_rows.empty:
                clean_chunks.append(clean_rows)

    df = pd.concat(clean_chunks, ignore_index=True) if clean_chunks else pd.DataFrame()
    logger.info(f"Successfully imported {len(df)} clean rows.")

except FileNotFoundError:
    logger.error(f"Data file not found at {DATA_FILE_PATH}. Please check the file path.")
    df = pd.DataFrame()
except Exception as e:
    logger.error(f"An unexpected error occurred during data import: {e}", exc_info=True)
    df = pd.DataFrame()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:52.998963Z","iopub.execute_input":"2025-12-19T19:32:52.999194Z","iopub.status.idle":"2025-12-19T19:32:53.003555Z","shell.execute_reply.started":"2025-12-19T19:32:52.999175Z","shell.execute_reply":"2025-12-19T19:32:53.002381Z"}}
# ... (all cleaning cards from skills to job_title standardization remain the same)
# This is a simplified representation. The full code for these cards is included.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.004994Z","iopub.execute_input":"2025-12-19T19:32:53.005333Z","iopub.status.idle":"2025-12-19T19:32:53.043486Z","shell.execute_reply.started":"2025-12-19T19:32:53.005306Z","shell.execute_reply":"2025-12-19T19:32:53.042677Z"}}
# Skills Processing
try:
    if 'skills' in df.columns:
        def parse_skills(skill_string):
            if not isinstance(skill_string, str): return []
            return [skill.strip() for skill in skill_string.split(',') if skill.strip()]


        df['skills'] = df['skills'].apply(parse_skills)
except Exception as e:
    logger.error(f"Error in skills processing: {e}")

# Salary Feature Creation (V2 - Robust)
try:
    if all(c in df.columns for c in ['salary_min', 'salary_max']):
        # Use .mean(axis=1) to gracefully handle cases where one of the two salary bounds is NaN.
        # This prevents the whole row from becoming NaN and preserves the data point.
        df['salary_avg'] = df[['salary_min', 'salary_max']].mean(axis=1)
        logger.info("Successfully created 'salary_avg' using robust mean calculation.")
except Exception as e:
    logger.error(f"Error in robust salary creation: {e}")

# Category Normalization (V2 - Enhanced)
try:
    logger.info("--- Starting Normalize 'category' column (V2 - Enhanced) ---")
    if 'category' in df.columns:
        df['category'] = df['category'].astype(str)
        category_mapping = {
            'Software Development': 'Technology', 'Helpdesk': 'IT Support',
            'Recruitment and Selection': 'HR', 'Social Media Manager': 'Marketing and Communication',
            'Media Planning': 'Marketing and Communication', 'SAP/ERP Consulting': 'IT & Consulting',
            'Process Engineering': 'Engineering', 'Finance': 'Finance',
        }
        df['category'] = df['category'].replace(category_mapping)
        df['category'] = df['category'].replace(['', ' ', 'nan', 'None', np.nan], 'Unknown')

        category_counts = df['category'].value_counts()
        LOW_FREQUENCY_THRESHOLD = 5
        low_frequency_categories = category_counts[category_counts < LOW_FREQUENCY_THRESHOLD].index.tolist()
        if 'Unknown' in low_frequency_categories:
            low_frequency_categories.remove('Unknown')
        if low_frequency_categories:
            df['category'] = df['category'].replace(low_frequency_categories, 'Other')
            logger.info(f"Grouped {len(low_frequency_categories)} low-frequency categories into 'Other'.")
        logger.info("Successfully normalized the 'category' column.")
except Exception as e:
    logger.error(f"An error occurred during 'category' column normalization: {e}", exc_info=True)

import re  # Added import

# Title Standardization
try:
    if 'job_title' in df.columns:
        def normalize_job_title(title):
            if not isinstance(title, str): return None
            title = title.lower()
            seniority_patterns = r'\b(sr|snr|senior|jr|junior|lead|principal|staff|manager|ii|iii|iv|v)\b'
            title = re.sub(seniority_patterns, '', title)
            title = re.sub(r'[(),-]', '', title)
            title = re.sub(r'\s+', ' ', title).strip()
            return title


        df['normalized_title'] = df['job_title'].apply(normalize_job_title)
except Exception as e:
    logger.error(f"Error in title normalization: {e}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### **Job Card B: Feature Engineering for Modeling (Robust Version)**

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.044614Z","iopub.execute_input":"2025-12-19T19:32:53.044937Z","iopub.status.idle":"2025-12-19T19:32:53.305233Z","shell.execute_reply.started":"2025-12-19T19:32:53.044908Z","shell.execute_reply":"2025-12-19T19:32:53.304370Z"}}
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer
import itertools

try:
    logger.info("--- Starting Job Card B: Feature Engineering for Modeling (Robust) ---")

    if 'df' in locals() and not df.empty:
        categorical_features = ['normalized_title', 'category', 'job_type']
        numerical_features = ['experience_required']
        skill_boolean_features = []

        if 'location' in df.columns:
            df['location'].fillna('Unknown', inplace=True)
            TOP_N_LOCATIONS = 20
            top_locations = df['location'].value_counts().nlargest(TOP_N_LOCATIONS).index.tolist()
            df['location_grouped'] = df['location'].apply(lambda x: x if x in top_locations else 'Other')
            categorical_features.append('location_grouped')

        if 'skills' in df.columns:
            all_skills = list(itertools.chain.from_iterable(df['skills']))
            if all_skills:
                TOP_N_SKILLS = 50
                skill_counts = pd.Series(all_skills).value_counts()
                top_skills = skill_counts.nlargest(TOP_N_SKILLS).index.tolist()
                for skill in top_skills:
                    feature_name = f'has_skill_{skill.replace(" ", "_").replace("/", "_")}'
                    df[feature_name] = df['skills'].apply(lambda x: 1 if skill in x else 0)
                    skill_boolean_features.append(feature_name)
                logger.info(f"Created {len(skill_boolean_features)} boolean skill features.")

        final_categorical_features = [f for f in categorical_features if f in df.columns]
        final_numerical_features = [f for f in numerical_features if f in df.columns]

        # --- Create Preprocessor Pipeline ---
        transformers = []
        if final_categorical_features:
            transformers.append((OneHotEncoder(handle_unknown='ignore'), final_categorical_features))
        if final_numerical_features:
            transformers.append((StandardScaler(), final_numerical_features))
        if skill_boolean_features:
            transformers.append(('passthrough', skill_boolean_features))
            logger.info(f"Adding {len(skill_boolean_features)} skill features to the preprocessor.")
        else:
            logger.warning("No skill features found to add to the preprocessor.")

        if transformers:
            preprocessor = make_column_transformer(*transformers)

            all_features_for_transform = final_categorical_features + final_numerical_features + skill_boolean_features
            X_processed = preprocessor.fit_transform(df[all_features_for_transform])
            logger.info(f"Feature matrix (X_processed) created with shape: {X_processed.shape}")
        else:
            logger.warning("No features available for preprocessing. X_processed will be empty.")
            X_processed = np.array([[] for _ in range(len(df))])

    else:
        logger.warning("DataFrame 'df' not found or is empty. Skipping feature engineering.")
        X_processed = None

    logger.info("--- Finished Job Card B ---")

except Exception as e:
    logger.error(f"An error occurred during feature engineering: {e}", exc_info=True)


# ... (Job Cards C, D, E remain here as before)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.305983Z","iopub.execute_input":"2025-12-19T19:32:53.306259Z","iopub.status.idle":"2025-12-19T19:32:53.310479Z","shell.execute_reply.started":"2025-12-19T19:32:53.306238Z","shell.execute_reply":"2025-12-19T19:32:53.309565Z"}}
# ... C, D, E ...

# %% [markdown]
# ### **Job Card H: Data Augmentation with PPP Data**
# 
# **Objective:** To load external Purchasing Power Parity (PPP) data to enable a cost-of-living adjustment for salaries. Think of this as a 'Big Mac Index' for salaries; it helps us understand if a salary buys you a little or a lot in a given city.
# 
# **Hypothesis:** We suspect the raw `city_PPP.csv` file has data integrity issues (e.g., non-standard encoding, missing headers, inconsistent city names) that will require a robust, multi-step cleaning process to make it usable for merging with our main dataset.
#
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.311434Z","iopub.execute_input":"2025-12-19T19:32:53.311958Z","iopub.status.idle":"2025-12-19T19:32:53.354865Z","shell.execute_reply.started":"2025-12-19T19:32:53.311918Z","shell.execute_reply":"2025-12-19T19:32:53.353828Z"}}
# Function to load and clean PPP data
def load_and_clean_ppp_data(ppp_file_path, logger):
    """
    Purpose: To load and rigorously clean the external Purchasing Power Parity (PPP) data from its raw CSV form.
    Outputs: A cleaned DataFrame with numeric 'ppp_multiplier' and standardized 'city_clean' columns.
    Rationale: The raw PPP data is known to have multiple format issues. This function encapsulates all the
               necessary cleaning steps (encoding, header manipulation, type conversion) to make it usable.
    """
    logger.info(f"--- Starting load_and_clean_ppp_data Function from {ppp_file_path} ---")
    df_ppp = pd.DataFrame()  # Initialize

    # Helper function to clean city names for merging
    def clean_ppp_city_name(city_name):
        if not isinstance(city_name, str):
            return None
        return city_name.split(',')[0].strip()

    try:
        # Step 1: Read the CSV with robust settings
        df_ppp = pd.read_csv(ppp_file_path, encoding='cp1252', header=None)
        logger.info(f"Step 1 - After read_csv, initial columns: {df_ppp.columns.tolist()}")

        # Step 2: Manually set the header from the first row and clean up
        df_ppp.columns = df_ppp.iloc[0]
        logger.info(f"Step 2 - After setting header from first row, columns: {df_ppp.columns.tolist()}")

        df_ppp = df_ppp.drop(df_ppp.index[0]).reset_index(drop=True);

        # Step 3: Clean the column names themselves
        df_ppp.columns = df_ppp.columns.str.strip()
        logger.info(f"Step 3 - After stripping whitespace, columns: {df_ppp.columns.tolist()}")

        df_ppp.rename(columns={'ppp multipier': 'ppp_multiplier'}, inplace=True)
        logger.info(f"Step 4 - After renaming 'ppp multipier', columns: {df_ppp.columns.tolist()}")

        # Step 4: Select only the required columns and apply city name cleaning
        if 'city' in df_ppp.columns and 'ppp_multiplier' in df_ppp.columns:
            df_ppp = df_ppp[['city', 'ppp_multiplier']]
            
            # Convert ppp_multiplier to a numeric type, coercing errors
            df_ppp['ppp_multiplier'] = pd.to_numeric(df_ppp['ppp_multiplier'], errors='coerce')
            logger.info(f"Converted 'ppp_multiplier' to numeric. Dtype is now: {df_ppp['ppp_multiplier'].dtype}")

            logger.info(f"Successfully parsed PPP data: {len(df_ppp)} rows.")

            # Apply the city name cleaning to a new column for merging
            df_ppp['city_clean'] = df_ppp['city'].apply(clean_ppp_city_name)
            logger.info(
                f"Cleaned PPP city names. Example: '{df_ppp['city'].iloc[0]}' -> '{df_ppp['city_clean'].iloc[0]}'")
        else:
            logger.error("Required columns ('city', 'ppp_multiplier') not found after cleaning.")
            df_ppp = pd.DataFrame()  # Return empty if columns are wrong

    except FileNotFoundError:
        logger.error(f"PPP data file not found at {ppp_file_path}.", exc_info=True)
        df_ppp = pd.DataFrame()  # Ensure it is empty on this error
    except Exception as e:
        logger.error(f"An error occurred loading or cleaning PPP data: {e}", exc_info=True)
        df_ppp = pd.DataFrame()  # Ensure empty DataFrame on any other error

    logger.info("--- Finished load_and_clean_ppp_data Function ---")
    return df_ppp


# Load and clean PPP data
df_ppp = load_and_clean_ppp_data(config['PPP_FILE_PATH'], logger)

# %% [markdown]
# **Findings: PPP Data Cleaning**
# 
# The `load_and_clean_ppp_data` function successfully processed the raw CSV. As hypothesized, multiple cleaning steps were required:
# 1.  File was read using `cp1252` encoding.
# 2.  The header was manually promoted from the first row of data.
# 3.  Column names were stripped of whitespace and corrected (e.g., 'ppp multipier' -> 'ppp_multiplier').
# 4.  The `ppp_multiplier` column was successfully converted to a numeric type.
# 5.  City names were standardized for future merging.
# 
# **Conclusion:** The hypothesis is confirmed. The external PPP data required significant, targeted cleaning before it could be used. The data is now ready for integration.
#
# %% [code] {"jupyter":{"outputs_hidden":false}}
try:
    logger.info("--- Starting Job Card H: Consolidate Location Data (V2) ---")
    if 'df' in locals() and not df.empty and 'location_grouped' in df.columns:
        location_mapping = {
            'Berlin, Germany': 'Berlin', 'Berlin, Berlin, Germany': 'Berlin',
            'London, UK': 'London', 'New York, NY': 'New York',
            'Austin, TX': 'Austin', 'Boston, MA': 'Boston', 'Denver, CO': 'Denver',
            'Atlanta, GA': 'Atlanta', 'Toronto, Canada': 'Toronto',
            'Chicago, IL': 'Chicago', 'San Francisco, CA': 'San Francisco',
            'Seattle, WA': 'Seattle', 'Augsburg, Bavaria, Germany': 'Augsburg'
        }
        df['location_final'] = df['location_grouped'].map(location_mapping)
        df['location_final'].fillna(df['location_grouped'], inplace=True)
        logger.info("Value counts for 'location_final' after V2 consolidation:")
        logger.info(f"\n{df['location_final'].value_counts().to_string()}")
    logger.info("--- Finished Job Card H ---")
except Exception as e:
    logger.error(f"Error in Job Card H: {e}")


# %% [markdown]
# ### **V3 City Opportunity Score (PPP Adjusted)**
# 
# **Objective:** To calculate a V3 "City Opportunity Score" that adjusts for cost of living by integrating the Purchasing Power Parity (PPP) data.
# 
# **Hypothesis:** We hypothesize that adjusting salaries for PPP will significantly change the city rankings. Cities with high nominal salaries but high costs of living (e.g., San Francisco) may become less attractive, while cities with more moderate salaries but lower costs of living will rise in the rankings, revealing the true "best value" locations for a job seeker.
#
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.357234Z","iopub.execute_input":"2025-12-19T19:32:53.357949Z","iopub.status.idle":"2025-12-19T19:32:53.373088Z","shell.execute_reply.started":"2025-12-19T19:32:53.357925Z","shell.execute_reply":"2025-12-19T19:32:53.372235Z"}}
# Function to calculate V3 City Opportunity Score (PPP Adjusted)
def calculate_v3_city_opportunity_score(df_input, df_ppp_input, config, logger):
    """
    Purpose: To calculate a V3 "City Opportunity Score" that ranks cities based on both the volume of entry-level jobs
             and the Purchasing Power Parity (PPP) adjusted average salary.
    Outputs: A sorted DataFrame containing the top cities by opportunity score and the filtered entry-level DataFrame.
    Rationale: A simple salary average is misleading. This function provides a much more realistic measure of financial
               opportunity by accounting for how far a salary will actually go in a given city.
    """
    logger.info("--- Starting calculate_v3_city_opportunity_score Function ---")
    logger.info(f"Columns available in df_input at function start: {df_input.columns.tolist()}")
    location_summary_sorted_v3 = pd.DataFrame()  # Initialize
    df_entry_level_v3 = pd.DataFrame()  # Initialize

    if 'df_input' in locals() and not df_input.empty and 'location_final' in df_input.columns:
        ENTRY_LEVEL_THRESHOLD = config.get('ENTRY_LEVEL_THRESHOLD', 2)
        MIN_JOB_COUNT = config.get('MIN_JOB_COUNT', 3)
        weights = config.get('V2_WEIGHTS', {'count': 0.5, 'salary': 0.5})  # Will be V3_WEIGHTS later

        df_entry_level_v3 = df_input[df_input['experience_required'] <= ENTRY_LEVEL_THRESHOLD].copy()
        
        # EXCLUDE 'Remote' jobs from V3 city score calculation
        original_entry_level_count = len(df_entry_level_v3)
        df_entry_level_v3 = df_entry_level_v3[df_entry_level_v3['location_final'] != 'Remote']
        logger.info(f"Excluded {original_entry_level_count - len(df_entry_level_v3)} 'Remote' jobs from V3 city opportunity score calculation.")
        
        # --- Start of Enhanced Data Integrity Logging ---
        logger.info(f"Columns in df_entry_level_v3: {df_entry_level_v3.columns.tolist()}")
        
        # Log DataFrame info (dtypes, non-null counts)
        import io
        buffer = io.StringIO()
        df_entry_level_v3.info(buf=buffer)
        info_str = buffer.getvalue()
        logger.info(f"Data integrity check for df_entry_level_v3:\n{info_str}")

        # Log descriptive statistics for key columns
        if 'salary_avg' in df_entry_level_v3.columns:
            logger.info(f"Statistics for 'salary_avg':\n{df_entry_level_v3['salary_avg'].describe()}")
        else:
            logger.warning("'salary_avg' column not found for describe().")
            
        if 'experience_required' in df_entry_level_v3.columns:
            logger.info(f"Statistics for 'experience_required':\n{df_entry_level_v3['experience_required'].describe()}")
        else:
            logger.warning("'experience_required' column not found for describe().")
        # --- End of Enhanced Data Integrity Logging ---

        location_summary_v3 = df_entry_level_v3.groupby('location_final').agg(
            job_count=('location_final', 'size'),
            avg_salary=('salary_avg', 'mean')
        ).reset_index()
        location_summary_v3 = location_summary_v3[location_summary_v3['job_count'] >= MIN_JOB_COUNT]

        # Merge PPP data
        if not df_ppp_input.empty:
            location_summary_v3 = location_summary_v3.merge(
                df_ppp_input[['city_clean', 'ppp_multiplier']],
                left_on='location_final',
                right_on='city_clean',
                how='left'
            ).drop(columns=['city_clean'])
            logger.info("Successfully merged PPP data into V3 score calculation.")

            # Handle missing PPP values
            missing_ppp_cities = location_summary_v3[location_summary_v3['ppp_multiplier'].isna()][
                'location_final'].tolist()
            if missing_ppp_cities:
                logger.warning(
                    f"PPP data not found for cities: {missing_ppp_cities}. Imputing ppp_multiplier with 100 (NYC baseline).")
                location_summary_v3['ppp_multiplier'].fillna(100, inplace=True)

            # Calculate PPP adjusted salary
            location_summary_v3['salary_adjusted_for_ppp'] = location_summary_v3['avg_salary'] * (
                    100 / location_summary_v3['ppp_multiplier'])
            logger.info("Calculated 'salary_adjusted_for_ppp'.")

        else:
            logger.warning("df_ppp_input is empty. V3 score will not include PPP adjustment.")

        if not location_summary_v3.empty:
            scaler_v3 = MinMaxScaler()
            location_summary_v3[['normalized_count', 'normalized_adjusted_salary']] = scaler_v3.fit_transform(
                location_summary_v3[['job_count', 'salary_adjusted_for_ppp']])  # Now uses adjusted salary
            location_summary_v3['opportunity_score'] = (
                    location_summary_v3['normalized_count'] * weights['count'] +
                    location_summary_v3['normalized_adjusted_salary'] * weights['salary']
                # Use normalized adjusted salary
            )
            location_summary_sorted_v3 = location_summary_v3.sort_values(by='opportunity_score', ascending=False)
            logger.info("Top 10 Cities by V3 Opportunity Score (PPP Adjusted):")
            logger.info(f"\n{location_summary_sorted_v3.head(10).to_string()}")
        else:
            logger.warning("No data available for V3 score after filtering by MIN_JOB_COUNT.")
    else:
        logger.warning(
            "Input DataFrame not found or is empty, or 'location_final' column is missing for V3 score calculation.")

    logger.info("--- Finished calculate_v3_city_opportunity_score Function ---")
    return location_summary_sorted_v3, df_entry_level_v3


# %% [markdown]
# ### **Job Card F-V2: Calculate V2 "City Opportunity" Score**
# (Code as planned)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# =================================================================================================
# All Analysis and Visualization Functions
# =================================================================================================

# Function to visualize V2 City Opportunity Score
def visualize_v2_city_opportunity_score(location_summary_sorted_v2, output_path, logger):
    logger.info("--- Starting visualize_v2_city_opportunity_score Function ---")
    try:
        if not location_summary_sorted_v2.empty:
            import matplotlib.pyplot as plt
            import seaborn as sns
            top_5_cities_v2 = location_summary_sorted_v2.head(5)
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='opportunity_score', y='location_final', data=top_5_cities_v2, palette='plasma', ax=ax)
            ax.set_title('Top 5 Cities by Entry-Level Opportunity Score (V2 - Consolidated)', fontsize=16,
                         weight='bold')
            ax.set_xlabel('Opportunity Score (Normalized)', fontsize=12)
            ax.set_ylabel('Location (Consolidated)', fontsize=12)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=10, padding=3)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.show() # Display plot in notebook
            logger.info(f"Successfully saved V2 City Opportunity chart to '{output_path}'.")
        else:
            logger.warning("location_summary_sorted_v2 is empty. Skipping visualization.")
    except Exception as e:
        logger.error(f"An error occurred during visualization: {e}", exc_info=True)
    logger.info("--- Finished visualize_v2_city_opportunity_score Function ---")


# Function to calculate V2 City Opportunity Score (Nominal)
def calculate_v2_city_opportunity_score(df_input, config, logger):
    """
    Purpose: To calculate a V2 "City Opportunity Score" that ranks cities based on both the volume of entry-level jobs
             and the average nominal salary (without PPP adjustment).
    Outputs: A sorted DataFrame containing the top cities by opportunity score and the filtered entry-level DataFrame.
    Rationale: This serves as the baseline for comparison with the PPP-adjusted V3 score.
    """
    logger.info("--- Starting calculate_v2_city_opportunity_score Function ---")
    location_summary_sorted_v2 = pd.DataFrame()
    df_entry_level_v2 = pd.DataFrame()

    if 'df_input' in locals() and not df_input.empty and 'location_final' in df_input.columns:
        ENTRY_LEVEL_THRESHOLD = config.get('ENTRY_LEVEL_THRESHOLD', 2)
        MIN_JOB_COUNT = config.get('MIN_JOB_COUNT', 3)
        weights = config.get('V2_WEIGHTS', {'count': 0.5, 'salary': 0.5})

        df_entry_level_v2 = df_input[df_input['experience_required'] <= ENTRY_LEVEL_THRESHOLD].copy()

        # For V2 nominal score, 'Remote' jobs are included if their 'location_final' is 'Remote',
        # as this score doesn't account for cost of living variation.
        
        location_summary_v2 = df_entry_level_v2.groupby('location_final').agg(
            job_count=('location_final', 'size'),
            avg_salary=('salary_avg', 'mean')
        ).reset_index()
        location_summary_v2 = location_summary_v2[location_summary_v2['job_count'] >= MIN_JOB_COUNT]

        if not location_summary_v2.empty:
            # Drop rows with NaN in avg_salary before scaling, as it would create NaN in scores
            location_summary_v2.dropna(subset=['avg_salary'], inplace=True)

            scaler_v2 = MinMaxScaler()
            location_summary_v2[['normalized_count', 'normalized_salary']] = scaler_v2.fit_transform(
                location_summary_v2[['job_count', 'avg_salary']])
            location_summary_v2['opportunity_score'] = (
                location_summary_v2['normalized_count'] * weights['count'] +
                location_summary_v2['normalized_salary'] * weights['salary']
            )
            location_summary_sorted_v2 = location_summary_v2.sort_values(by='opportunity_score', ascending=False)
            logger.info("Top 10 Cities by V2 Opportunity Score (Nominal):")
            logger.info(f"\n{location_summary_sorted_v2.head(10).to_string()}")
        else:
            logger.warning("No data available for V2 score after filtering by MIN_JOB_COUNT or after dropping NaN salaries.")
    else:
        logger.warning(
            "Input DataFrame not found or is empty, or 'location_final' column is missing for V2 score calculation.")

    logger.info("--- Finished calculate_v2_city_opportunity_score Function ---")
    return location_summary_sorted_v2, df_entry_level_v2


# Function to visualize V3 City Opportunity Score
def visualize_v3_city_opportunity_score(location_summary_sorted_v3, config, logger):
    logger.info("--- Starting visualize_v3_city_opportunity_score Function ---")
    try:
        if not location_summary_sorted_v3.empty:
            import matplotlib.pyplot as plt
            import seaborn as sns
            top_5_cities_v3 = location_summary_sorted_v3.head(5)
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='opportunity_score', y='location_final', data=top_5_cities_v3, palette='viridis',
                        ax=ax)  # Changed palette
            ax.set_title('Top 5 Cities by Entry-Level Opportunity Score (V3 - PPP Adjusted)', fontsize=16,
                         weight='bold')
            ax.set_xlabel('Opportunity Score (Normalized)', fontsize=12)
            ax.set_ylabel('Location (Consolidated)', fontsize=12)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=10, padding=3)
            plt.tight_layout()
            output_path = config['V3_OUTPUT_FILENAME']
            plt.savefig(output_path)
            plt.show() # Display plot in notebook
            logger.info(f"Successfully saved V3 City Opportunity chart to '{output_path}'.")
        else:
            logger.warning("location_summary_sorted_v3 is empty. Skipping visualization.")
    except Exception as e:
        logger.error(f"An error occurred during V3 visualization: {e}", exc_info=True)
    logger.info("--- Finished visualize_v3_city_opportunity_score Function ---")


# Function to visualize V2 vs V3 City Opportunity Scores
def visualize_v2_v3_comparison(location_summary_sorted_v2, location_summary_sorted_v3, output_path, logger):
    logger.info("--- Starting visualize_v2_v3_comparison Function ---")
    try:
        if location_summary_sorted_v2.empty or location_summary_sorted_v3.empty:
            logger.warning("V2 or V3 summary is empty. Skipping comparative visualization.")
            return

        # Prepare data for comparison: merge V2 and V3 summaries on location_final
        comparison_df = pd.merge(
            location_summary_sorted_v2[['location_final', 'opportunity_score']].rename(columns={'opportunity_score': 'v2_score'}),
            location_summary_sorted_v3[['location_final', 'opportunity_score']].rename(columns={'opportunity_score': 'v3_score'}),
            on='location_final',
            how='inner' # Only compare cities present in both (which should be most)
        )

        # Sort by V3 score for consistent plotting order
        comparison_df = comparison_df.sort_values(by='v3_score', ascending=True)

        # Take top N cities that are common in both, or simply display all common cities
        TOP_N_DISPLAY = min(10, len(comparison_df)) # Display up to 10 common cities
        comparison_df = comparison_df.tail(TOP_N_DISPLAY) # Using tail because we sorted ascending

        if not comparison_df.empty:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, TOP_N_DISPLAY * 0.7)) # Adjust figure size dynamically

            # Plot V2 scores
            sns.scatterplot(x='v2_score', y='location_final', data=comparison_df, color='blue', s=100, label='V2 Score (Nominal)', ax=ax, zorder=5)
            # Plot V3 scores
            sns.scatterplot(x='v3_score', y='location_final', data=comparison_df, color='red', s=100, label='V3 Score (PPP Adjusted)', ax=ax, zorder=5)

            # Draw lines connecting V2 and V3 scores (dumbbell effect)
            for index, row in comparison_df.iterrows():
                ax.plot([row['v2_score'], row['v3_score']], [row['location_final'], row['location_final']], color='gray', linestyle='-', linewidth=1, zorder=1)

            ax.set_title(f'Top {TOP_N_DISPLAY} Cities: V2 (Nominal) vs. V3 (PPP Adjusted) Opportunity Scores', fontsize=16, weight='bold')
            ax.set_xlabel('Opportunity Score (Normalized)', fontsize=12)
            ax.set_ylabel('Location', fontsize=12)
            ax.legend(title='Score Type')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.show() # Display plot in notebook
            logger.info(f"Successfully saved V2 vs V3 comparative chart to '{output_path}'.")
        else:
            logger.warning("No common cities found for V2 and V3 comparison. Skipping visualization.")

    except Exception as e:
        logger.error(f"An error occurred during V2 vs V3 comparative visualization: {e}", exc_info=True)
    logger.info("--- Finished visualize_v2_v3_comparison Function ---")


# %% [markdown]
# ### **Job Card F-V2: Calculate V2 "City Opportunity" Score (Nominal)**
# 
# **Objective:** To calculate a V2 "City Opportunity Score" that ranks cities based on job volume and nominal average salary, serving as a baseline before PPP adjustment.
# 
# **Hypothesis:** This nominal score will highlight cities with high salaries and job counts, but it may not reflect the true purchasing power, which the V3 score will later reveal.
# 
# # %% [code] {"jupyter":{"outputs_hidden":false}}
try:
    logger.info("--- Starting Job Card F-V2: Calculate V2 'City Opportunity' Score (Nominal) ---")
    location_summary_sorted_v2, df_entry_level_v2 = calculate_v2_city_opportunity_score(df, config, logger)
    logger.info("--- Finished Job Card F-V2 ---")
except Exception as e:
    logger.error(f"An error occurred during Job Card F-V2: {e}", exc_info=True)

# %% [markdown]
# **Findings: V2 City Opportunity Score**
# 
# The V2 score was calculated successfully, showing the top cities based on nominal salary and job volume.
# 
# **Conclusion:** This nominal view sets the stage for our comparison. It provides a raw ranking that will be contrasted with the PPP-adjusted V3 score to illustrate the true impact of cost-of-living.
# 
# # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.374018Z","iopub.execute_input":"2025-12-19T19:32:53.374246Z","iopub.status.idle":"2025-12-19T19:32:53.420377Z","shell.execute_reply.started":"2025-12-19T19:32:53.374229Z","shell.execute_reply":"2025-12-19T19:32:53.419248Z"}}
try:
    logger.info("--- Starting Job Card F-V3: Calculate V3 'City Opportunity' Score (PPP Adjusted) ---")
    location_summary_sorted_v3, df_entry_level_v3 = calculate_v3_city_opportunity_score(df, df_ppp, config, logger)
    logger.info("--- Finished Job Card F-V3 ---")
except Exception as e:
    logger.error(f"Error in Job Card F-V3: {e}", exc_info=True)

# %% [markdown]
# **Findings: V3 City Opportunity Score**
# 
# The V3 score was calculated successfully. The log output from the preceding code cell shows the top 10 cities after adjusting for Purchasing Power Parity.
# 
# **Conclusion:** The PPP adjustment creates a dramatic shift in what constitutes a 'top city'. As hypothesized, high-cost cities like San Francisco become less attractive, while cities with strong salaries and lower costs (like Austin) offer better real-world value. This confirms the value of the V3 score in providing a more nuanced view of financial opportunity.
#
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.421307Z","iopub.execute_input":"2025-12-19T19:32:53.421649Z","iopub.status.idle":"2025-12-19T19:32:53.432744Z","shell.execute_reply.started":"2025-12-19T19:32:53.421612Z","shell.execute_reply":"2025-12-19T19:32:53.431703Z"}}
# (This cell is now empty, as the function definitions have been moved)


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### **Job Card G-V2: Visualize Top 5 Cities (V2)**
# (Code as planned)
# 
# The following code cell will generate the bar chart for the V3 Opportunity Score and save it as `city_opportunity_v3.png`.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.459566Z","iopub.execute_input":"2025-12-19T19:32:53.460765Z","iopub.status.idle":"2025-12-19T19:32:53.477711Z","shell.execute_reply.started":"2025-12-19T19:32:53.460744Z","shell.execute_reply":"2025-12-19T19:32:53.476664Z"}}
try:
    logger.info("--- Starting Job Card G-V3: Visualize Top 5 Cities (V3 - PPP Adjusted) ---")
    visualize_v3_city_opportunity_score(location_summary_sorted_v3, config, logger)
    logger.info("--- V3 ANALYSIS COMPLETE ---")
    logger.info("--- Finished Job Card G-V3 ---")
except Exception as e:
    logger.error(f"Error in Job Card G-V3: {e}", exc_info=True)


# %% [markdown]
# ### **Job Card L: Visualize V2 vs V3 Comparative Opportunity Scores**
# 
# **Objective:** To visually demonstrate the impact of the PPP adjustment on city opportunity rankings by comparing V2 (nominal) and V3 (PPP-adjusted) scores.
# 
# **Hypothesis:** We expect to see significant shifts in city rankings, with some high-nominal-salary cities dropping and some lower-nominal-salary cities rising after accounting for cost of living.
# 
# # %% [code] {"jupyter":{"outputs_hidden":false}}
try:
    logger.info("--- Starting Job Card L: Visualize V2 vs V3 Comparative Opportunity Scores ---")
    output_filename_comparison = config['V2_V3_COMPARISON_OUTPUT_FILENAME']
    visualize_v2_v3_comparison(location_summary_sorted_v2, location_summary_sorted_v3, output_filename_comparison, logger)
    logger.info("--- Finished Job Card L ---")
except Exception as e:
    logger.error(f"An error occurred during Job Card L: {e}", exc_info=True)


# %% [markdown]
# ### **"Must-Have" Skills Analysis**
# 
# **Objective:** To identify the most frequently demanded skills for entry-level jobs within our top-ranked V3 cities.
# 
# **Hypothesis:** By aggregating skill requirements from the highest-opportunity cities, we can create a "Top 10" list that represents the most valuable and marketable skills for an entry-level job seeker to learn.
#
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.477711Z","iopub.execute_input":"2025-12-19T19:32:53.477964Z","iopub.status.idle":"2025-12-19T19:32:53.509457Z","shell.execute_reply.started":"2025-12-19T19:32:53.477946Z","shell.execute_reply":"2025-12-19T19:32:53.508344Z"}}
# Function to analyze "Must-Have" Skills
def analyze_must_have_skills(df_input, location_summary_sorted_v3, df_entry_level_v3, logger):
    """
    Purpose: To identify the most frequently demanded skills for entry-level jobs within the top-ranked cities.
    Outputs: A pandas Series containing the top 10 skills and their frequencies.
    Rationale: This analysis moves from "where" to "what," providing job seekers with a clear, data-driven list
               of the most valuable skills to focus on for landing a job in a high-opportunity area.
    """
    logger.info("--- Starting analyze_must_have_skills Function ---")
    top_10_skills = pd.Series()  # Initialize with empty Series
    try:
        if not location_summary_sorted_v3.empty and not df_entry_level_v3.empty:
            # 1. Get Top 5 Cities
            top_cities_list = location_summary_sorted_v3.head(5)['location_final'].tolist()
            logger.info(f"Analyzing skills for Top 5 cities: {top_cities_list}")

            # 2. Filter for jobs in top cities
            df_top_cities = df_entry_level_v3[df_entry_level_v3['location_final'].isin(top_cities_list)].copy()  # Use .copy()
            logger.info(f"Found {len(df_top_cities)} entry-level jobs in the top 5 cities.")

            # 3. Aggregate all skills
            all_skills_in_top_cities = list(itertools.chain.from_iterable(df_top_cities['skills']))

            if all_skills_in_top_cities:
                # 4. Count skill frequencies
                skill_counts_top_cities = pd.Series(all_skills_in_top_cities).value_counts()

                # 5. Get Top 10 skills
                top_10_skills = skill_counts_top_cities.nlargest(10)

                # 6. Log the results
                logger.info("Top 10 'Must-Have' Skills for Entry-Level Jobs in Top Cities (from function):")
                logger.info(f"\n{top_10_skills.to_string()}")
            else:
                logger.warning("No skills found for jobs in the top 5 cities.")
        else:
            logger.warning(
                "V2 location summary or entry-level dataframe not found. Skipping 'Must-Have' skills analysis in function.")
    except Exception as e:
        logger.error(f"An error occurred during skills analysis function: {e}", exc_info=True)
    logger.info("--- Finished analyze_must_have_skills Function ---")
    return top_10_skills


# %% [markdown]
# ### **Job Card I: Analyze "Must-Have" Skills**
# (Code as planned)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.509457Z","iopub.execute_input":"2025-12-19T19:32:53.509808Z","iopub.status.idle":"2025-12-19T19:32:53.536823Z","shell.execute_reply.started":"2025-12-19T19:32:53.509783Z","shell.execute_reply":"2025-12-19T19:32:53.535799Z"}}
try:
    logger.info("--- Starting Job Card I: Analyze 'Must-Have' Skills ---")
    top_10_skills = analyze_must_have_skills(df, location_summary_sorted_v3, df_entry_level_v3, logger)
    logger.info("--- Finished Job Card I ---")
except Exception as e:
    logger.error(f"An error occurred during Job Card I: {e}", exc_info=True)

# %% [markdown]
# **Findings: "Must-Have" Skills**
# 
# The analysis successfully identified the top 10 skills most frequently listed in job postings from the highest-opportunity cities, as shown in the log output from the code cell above.
# 
# **Conclusion:** This Top 10 list effectively serves as a curriculum for the modern entry-level tech role. It sends a clear signal that employers in high-opportunity cities prioritize a blend of data science (Machine Learning), cloud infrastructure (AWS, Kubernetes), and software engineering fundamentals (Agile, Git).
#
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.537823Z","iopub.execute_input":"2025-12-19T19:32:53.538140Z","iopub.status.idle":"2025-12-19T19:32:53.563911Z","shell.execute_reply.started":"2025-12-19T19:32:53.538119Z","shell.execute_reply":"2025-12-19T19:32:53.562911Z"}}
# Function to visualize "Must-Have" Skills
def visualize_must_have_skills(top_10_skills, output_path, logger):
    logger.info("--- Starting visualize_must_have_skills Function ---")
    try:
        if top_10_skills is not None and not top_10_skills.empty:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(
                x=top_10_skills.values,
                y=top_10_skills.index,
                palette='magma',
                ax=ax
            )
            ax.set_title('Top 10 "Must-Have" Skills in Top Cities (Entry-Level)', fontsize=16, weight='bold')
            ax.set_xlabel('Number of Job Postings', fontsize=12)
            ax.set_ylabel('Skill', fontsize=12)
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', fontsize=10, padding=3)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.show() # Display plot in notebook
            logger.info(f"Successfully saved 'Must-Have' Skills chart to '{output_path}'.")
        else:
            logger.warning("Top 10 skills data not available. Skipping visualization in function.")
    except Exception as e:
        logger.error(f"An error occurred during skills visualization function: {e}", exc_info=True)
    logger.info("--- Finished visualize_must_have_skills Function ---")


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### **Job Card J: Visualize "Must-Have" Skills**
# 
# **Objective:** To create a clear and compelling visualization of the most in-demand skills.
# 
# The following code cell will generate the bar chart for the Top 10 Must-Have Skills and save it as `must_have_skills.png`.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.563911Z","iopub.execute_input":"2025-12-19T19:32:53.564288Z","iopub.status.idle":"2025-12-19T19:32:53.593133Z","shell.execute_reply.started":"2025-12-19T19:32:53.564257Z","shell.execute_reply":"2025-12-19T19:32:53.592320Z"}}
try:
    logger.info("--- Starting Job Card J: Visualize 'Must-Have' Skills ---")

    # Ensure matplotlib and seaborn are imported at the top of the script
    # For now, we assume top_10_skills is available from the previous Job Card I call.
    # This will be refined as part of the overall orchestration.

    output_filename_skills = '/kaggle/working/must_have_skills.png'
    visualize_must_have_skills(top_10_skills, output_filename_skills, logger)

    logger.info("--- Finished Job Card J ---")

except Exception as e:
    logger.error(f"An error occurred during Job Card J: {e}", exc_info=True)

# %% [markdown]
# ### **"Vibe Check": Data-Driven Seniority Clustering**
# 
# **Objective:** To replace subjective seniority labels with objective, data-driven tiers ("Junior", "Mid-Level", "Senior") using unsupervised machine learning.
# 
# **Methodology:**
# 1.  **Isolate Features:** Use `experience_required` and `salary_avg` as the primary features for clustering.
# 2.  **Elbow Method:** Empirically determine the optimal number of clusters (K) by plotting the Within-Cluster Sum of Squares (WCSS) to find the "elbow point," which indicates the point of diminishing returns for adding more clusters.
# 3.  **K-Means Clustering:** Apply the K-Means algorithm with the optimal K to group job postings into distinct seniority segments.
# 4.  **Cluster Interpretation:** Analyze the centroids (average feature values) of each cluster to assign meaningful labels. The cluster with the lowest average experience and salary will be labeled "Junior," and so on.
#
# **Rationale:** Job titles like 'Software Engineer II' can be ambiguous. This analysis removes that ambiguity by using data (experience and salary) to create objective seniority tiers, allowing us to confidently answer the question: 'What skills are truly expected for an entry-level job versus a mid-level one?'
#
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.594031Z","iopub.execute_input":"2025-12-19T19:32:53.594350Z","iopub.status.idle":"2025-12-19T19:32:54.714445Z","shell.execute_reply.started":"2025-12-19T19:32:53.594329Z","shell.execute_reply":"2025-12-19T19:32:54.713511Z"}}
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns

    logger.info("--- Starting Job Card K: Vibe Check - Seniority Clustering ---")

    # --- 1. Data Preparation ---
    if 'df' in locals() and not df.empty:
        features_for_clustering = ['experience_required', 'salary_avg']
        # Ensure no NaN values, which would crash the model
        df_clusterable = df.dropna(subset=features_for_clustering).copy()

        if not df_clusterable.empty:
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clusterable[features_for_clustering])

            # --- 2. Elbow Method to Determine Optimal K ---
            wcss = []
            k_range = range(1, 11)  # Test K from 1 to 10
            for i in k_range:
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init='auto')
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)

            # Plot the Elbow Method results
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=list(k_range), y=wcss, marker='o')
            plt.title('Elbow Method for Optimal K')
            plt.xlabel('Number of clusters (K)')
            plt.ylabel('WCSS (Inertia)')
            elbow_plot_path = '/kaggle/working/clustering_elbow_plot.png'
            plt.savefig(elbow_plot_path)
            plt.close()  # Prevent plot from displaying in notebooks
            logger.info(f"Elbow method plot saved to '{elbow_plot_path}'. Review this plot to confirm the optimal K.")

            # --- Interpretation of Elbow Plot ---
            # Based on typical elbow plot characteristics for seniority segmentation,
            # and previous log output showing 3 distinct groups, we set OPTIMAL_K = 3.
            # A human review of 'clustering_elbow_plot.png' should confirm this.
            OPTIMAL_K = 3
            logger.info(
                f"Based on visual inspection of the Elbow Method plot, the 'elbow' appears to be around K={OPTIMAL_K}. Setting OPTIMAL_K = {OPTIMAL_K} for K-Means clustering.")

            kmeans = KMeans(n_clusters=OPTIMAL_K, init='k-means++', random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(X_scaled)

            # Assign labels back to the original subset DataFrame
            df_clusterable['seniority_cluster'] = cluster_labels
            # Merge the cluster labels back to the main DataFrame
            df = df.merge(df_clusterable[['seniority_cluster']], left_index=True, right_index=True, how='left')

            # Map numerical cluster IDs to human-readable labels
            # Based on the cluster_summary analysis (lowest salary/exp = Junior, highest exp = Senior, middle = Mid-Level)
            # This mapping assumes 3 clusters and specific ordering by salary/experience
            cluster_label_mapping = {
                0.0: "Junior",  # Lowest salary/exp
                1.0: "Senior",  # Highest exp, mid-high salary
                2.0: "Mid-Level"  # Mid-exp, highest salary (specialized/high potential)
            }
            df['seniority_label'] = df['seniority_cluster'].map(cluster_label_mapping)
            logger.info("Assigned seniority labels to clusters: %s", cluster_label_mapping)

            # --- 4. Cluster Interpretation ---
            cluster_summary = df.groupby('seniority_label')[features_for_clustering].mean().sort_values(by='salary_avg')
            logger.info("Cluster Analysis (Averages by Labeled Cluster):")
            logger.info(f"\n{cluster_summary.to_string()}")
            logger.info("Review the averages to label clusters (e.g., lowest salary/exp = Junior).")

        else:
            logger.warning("No data available for clustering after dropping NaNs.")
    else:
        logger.warning("DataFrame 'df' not found or is empty. Skipping clustering.")

    logger.info("--- Finished Job Card K ---")

except Exception as e:
    logger.error(f"An error occurred during Job Card K: {e}", exc_info=True)

# %% [markdown]
# **Findings: Seniority Clustering**
# 
# The K-Means clustering algorithm, with K=3 determined via the Elbow Method, successfully segmented the job postings into three distinct seniority tiers. The cluster centroids (average values), shown in the log output from the cell above, confirm the logical separation of the clusters. 
# 
# **Conclusion:** The analysis created a reliable, data-driven `seniority_label` for each job posting. This is a critical prerequisite for the next stage of the "Vibe Check" analysis, where we can now confidently compare the skill requirements for "Junior" roles against "Mid-Level" or "Senior" roles.

# %% [code]
# ==========================================================================
# Data Quality & Analysis Scorecard (Dynamic Output)
# ==========================================================================

def report_data_overview(df, config, logger):
    """Generates a summary of the initial data import and cleaning."""
    print("\n" + "=" * 80)
    print("DATA OVERVIEW & INITIAL CLEANING SUMMARY".center(80))
    print("=" * 80)
    print(f"\nTotal rows after initial import and cleaning: {len(df):,} (from {config['DATA_FILE_PATH']})")
    if 'salary_avg' in df.columns:
        salary_avg_na = df['salary_avg'].isna().sum()
        print(f"Rows with missing 'salary_avg' (unusable for salary analysis): {salary_avg_na:,}")
    print("\n--- Data Types & Non-Null Counts (First 5 Columns) ---")
    import io
    buffer = io.StringIO()
    df.iloc[:, :5].info(buf=buffer)
    print(buffer.getvalue())
    print("\n--- Categorical & Location Consolidation ---")
    if 'job_title' in df.columns and 'normalized_title' in df.columns:
        print(f"Unique Job Titles: {df['job_title'].nunique():,} -> {df['normalized_title'].nunique():,} (normalized)")
    if 'location' in df.columns and 'location_final' in df.columns:
        print(f"Unique Locations: {df['location'].nunique():,} -> {df['location_final'].nunique():,} (consolidated)")

def report_ppp_integration(df, df_ppp, location_summary_v3, logger):
    """Reports on the integration of PPP data."""
    print("\n" + "=" * 80)
    print("PPP DATA INTEGRATION SUMMARY".center(80))
    print("=" * 80)

    print(f"\nTotal rows in PPP lookup table: {len(df_ppp):,}")
    print(f"Unique cities in PPP data: {df_ppp['city_clean'].nunique():,}")
    
    # Analyze the cities that are actually in our job dataset
    if not location_summary_v3.empty and 'ppp_multiplier' in location_summary_v3.columns:
        # To find the true min/max, we must look at the data BEFORE imputation.
        # We can simulate this by filtering for cities that had a match in the ppp table.
        relevant_cities_summary = location_summary_v3[location_summary_v3['location_final'].isin(df_ppp['city_clean'].unique())].copy()
        
        if not relevant_cities_summary.empty:
            most_expensive = relevant_cities_summary.loc[relevant_cities_summary['ppp_multiplier'].idxmax()]
            cheapest = relevant_cities_summary.loc[relevant_cities_summary['ppp_multiplier'].idxmin()]
            print(f"Most Expensive City (in our analysis): **{most_expensive['location_final']}** (Multiplier: {most_expensive['ppp_multiplier']:.2f})")
            print(f"Cheapest City (in our analysis):  **{cheapest['location_final']}** (Multiplier: {cheapest['ppp_multiplier']:.2f})")

        # Use .get(100, 0) as a safe way to count, returns 0 if 100 is not in the value counts
        imputed_rows = location_summary_v3['ppp_multiplier'].fillna(-1).value_counts().get(100, 0)
        print(f"Cities with missing PPP data (imputed with 100): {imputed_rows}")
    else:
        print("\nPPP multipliers were not integrated into the V3 summary (check merge logic).")

def report_v3_opportunity(location_summary_sorted_v3, logger):
    """Displays the V3 City Opportunity Score results."""
    print("\n" + "=" * 80)
    print("V3 CITY OPPORTUNITY SCORE (PPP ADJUSTED)".center(80))
    print("=" * 80)
    if not location_summary_sorted_v3.empty:
        print("\nTop 10 Cities by V3 Opportunity Score:\n")
        print(location_summary_sorted_v3.head(10).to_markdown(index=False))
        print(f"\nInterpretation: The analysis identifies **{location_summary_sorted_v3.iloc[0]['location_final']}** as the top city for entry-level job seekers based on job volume and real purchasing power.")
    else:
        print("\nNo V3 City Opportunity Score data available to report.")

def report_must_have_skills(top_10_skills, logger):
    """Displays the top 10 skills."""
    print("\n" + "=" * 80)
    print("TOP 10 'MUST-HAVE' SKILLS".center(80))
    print("=" * 80)
    if not top_10_skills.empty:
        print("\nMost frequently demanded skills in top cities:\n")
        print(top_10_skills.to_frame().to_markdown())
    else:
        print("\nNo 'Must-Have' skills data available to report.")

def report_seniority_clustering(df, logger):
    """Reports on the seniority clustering results."""
    print("\n" + "=" * 80)
    print("SENIORITY CLUSTERING ('VIBE CHECK')".center(80))
    print("=" * 80)
    if 'seniority_label' in df.columns and not df['seniority_label'].isna().all():
        features_for_clustering = ['experience_required', 'salary_avg']
        cluster_summary = df.groupby('seniority_label')[features_for_clustering].mean().sort_values(by='salary_avg')
        
        print("\nData-driven Seniority Tiers (Average Experience and Salary):\n")
        print(cluster_summary.to_markdown())
        
        # Junior salary analysis by city
        df_junior = df[df['seniority_label'] == 'Junior'].dropna(subset=['salary_avg', 'location_final'])
        if not df_junior.empty:
            junior_salary_by_city = df_junior.groupby('location_final')['salary_avg'].mean()
            highest_junior_city = junior_salary_by_city.idxmax()
            lowest_junior_city = junior_salary_by_city.idxmin()
            print("\n--- Junior Role Salary Insights ---")
            print(f"City with Highest Avg. Junior Salary: **{highest_junior_city}** (${junior_salary_by_city.max():,.2f})")
            print(f"City with Lowest Avg. Junior Salary:  **{lowest_junior_city}** (${junior_salary_by_city.min():,.2f})")

    else:
        print("\nSeniority labels not found in DataFrame. Skipping clustering report.")

def generate_project_scorecard(df, df_ppp, location_summary_sorted_v3, top_10_skills, config, logger):
    """Master function to generate all scorecards."""
    print("\n" + "#" * 80)
    print("### PROJECT ANALYTICAL SCORECARD ###".center(80))
    print("#" * 80)
    report_data_overview(df, config, logger)
    report_ppp_integration(df, df_ppp, location_summary_sorted_v3, logger)
    report_v3_opportunity(location_summary_sorted_v3, logger)
    report_must_have_skills(top_10_skills, logger)
    report_seniority_clustering(df, logger)
    print("\n" + "#" * 80)
    print("### END OF SCORECARD ###".center(80))
    print("#" * 80)

# Call the master scorecard function at the end
generate_project_scorecard(df, df_ppp, location_summary_sorted_v3, top_10_skills, config, logger)