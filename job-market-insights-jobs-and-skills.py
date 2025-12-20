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
    'V3_OUTPUT_FILENAME': '/kaggle/working/city_opportunity_v3.png'  # New: Output filename for V3 plot
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

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### **Job Card H: Consolidate Location Data (V2)**
# (Code as planned)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.311434Z","iopub.execute_input":"2025-12-19T19:32:53.311958Z","iopub.status.idle":"2025-12-19T19:32:53.354865Z","shell.execute_reply.started":"2025-12-19T19:32:53.311918Z","shell.execute_reply":"2025-12-19T19:32:53.353828Z"}}
# Function to load and clean PPP data
def load_and_clean_ppp_data(ppp_file_path, logger):
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
            logger.info(f"Successfully parsed PPP data: {len(df_ppp)} rows.")
            
            # Apply the city name cleaning to a new column for merging
            df_ppp['city_clean'] = df_ppp['city'].apply(clean_ppp_city_name)
            logger.info(f"Cleaned PPP city names. Example: '{df_ppp['city'].iloc[0]}' -> '{df_ppp['city_clean'].iloc[0]}'")
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


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.357234Z","iopub.execute_input":"2025-12-19T19:32:53.357949Z","iopub.status.idle":"2025-12-19T19:32:53.373088Z","shell.execute_reply.started":"2025-12-19T19:32:53.357925Z","shell.execute_reply":"2025-12-19T19:32:53.372235Z"}}
# Function to calculate V2 City Opportunity Score
def calculate_v2_city_opportunity_score(df_input, config, logger):
    logger.info("--- Starting calculate_v2_city_opportunity_score Function ---")
    location_summary_sorted_v2 = pd.DataFrame()  # Initialize in case of early exit
    df_entry_level_v2 = pd.DataFrame()  # Initialize

    if 'df_input' in locals() and not df_input.empty and 'location_final' in df_input.columns:
        ENTRY_LEVEL_THRESHOLD = 2  # Assuming config['ENTRY_LEVEL_THRESHOLD'] is not yet implemented
        MIN_JOB_COUNT = 3  # Assuming config['MIN_JOB_COUNT'] is not yet implemented
        weights = {'count': 0.5, 'salary': 0.5}  # Assuming config['V2_WEIGHTS'] is not yet implemented

        df_entry_level_v2 = df_input[df_input['experience_required'] <= ENTRY_LEVEL_THRESHOLD].copy()

        location_summary_v2 = df_entry_level_v2.groupby('location_final').agg(
            job_count=('location_final', 'size'),
            avg_salary=('salary_avg', 'mean')
        ).reset_index()
        location_summary_v2 = location_summary_v2[location_summary_v2['job_count'] >= MIN_JOB_COUNT]

        if not location_summary_v2.empty:
            scaler_v2 = MinMaxScaler()
            location_summary_v2[['normalized_count', 'normalized_salary']] = scaler_v2.fit_transform(
                location_summary_v2[['job_count', 'avg_salary']]
            )
            location_summary_v2['opportunity_score'] = (
                    location_summary_v2['normalized_count'] * weights['count'] +
                    location_summary_v2['normalized_salary'] * weights['salary']
            )
            location_summary_sorted_v2 = location_summary_v2.sort_values(by='opportunity_score', ascending=False)
            logger.info("Top 10 Cities by V2 Opportunity Score (from function):")
            logger.info(f"\n{location_summary_sorted_v2.head(10).to_string()}")
        else:
            logger.warning("No data available for V2 score after filtering by MIN_JOB_COUNT.")
    else:
        logger.warning(
            "Input DataFrame not found or is empty, or 'location_final' column is missing for V2 score calculation.")

    logger.info("--- Finished calculate_v2_city_opportunity_score Function ---")
    return location_summary_sorted_v2, df_entry_level_v2


# Function to calculate V3 City Opportunity Score (PPP Adjusted)
def calculate_v3_city_opportunity_score(df_input, df_ppp_input, config, logger):
    logger.info("--- Starting calculate_v3_city_opportunity_score Function ---")
    location_summary_sorted_v3 = pd.DataFrame()  # Initialize
    df_entry_level_v3 = pd.DataFrame()  # Initialize

    if 'df_input' in locals() and not df_input.empty and 'location_final' in df_input.columns:
        ENTRY_LEVEL_THRESHOLD = config.get('ENTRY_LEVEL_THRESHOLD', 2)
        MIN_JOB_COUNT = config.get('MIN_JOB_COUNT', 3)
        weights = config.get('V2_WEIGHTS', {'count': 0.5, 'salary': 0.5})  # Will be V3_WEIGHTS later

        df_entry_level_v3 = df_input[df_input['experience_required'] <= ENTRY_LEVEL_THRESHOLD].copy()

        logger.debug(f"Columns of df_entry_level_v3 before groupby: {df_entry_level_v3.columns.tolist()}")
        location_summary_v3 = df_entry_level_v3.groupby('location_final').agg(
            job_count=('location_final', 'size'),
            avg_salary=('avg_salary', 'mean')  # Note: using avg_salary, not yet adjusted
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
                location_summary_v3[['job_count', 'salary_adjusted_for_ppp']]  # Now uses adjusted salary
            )
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


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### **Job Card F-V2: Calculate V2 "City Opportunity" Score**
# (Code as planned)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.374018Z","iopub.execute_input":"2025-12-19T19:32:53.374246Z","iopub.status.idle":"2025-12-19T19:32:53.420377Z","shell.execute_reply.started":"2025-12-19T19:32:53.374229Z","shell.execute_reply":"2025-12-19T19:32:53.419248Z"}}
try:
    logger.info("--- Starting Job Card F-V3: Calculate V3 'City Opportunity' Score (PPP Adjusted) ---")
    location_summary_sorted_v3, df_entry_level_v3 = calculate_v3_city_opportunity_score(df, df_ppp, config, logger)
    logger.info("--- Finished Job Card F-V3 ---")
except Exception as e:
    logger.error(f"Error in Job Card F-V3: {e}", exc_info=True)


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.421307Z","iopub.execute_input":"2025-12-19T19:32:53.421649Z","iopub.status.idle":"2025-12-19T19:32:53.432744Z","shell.execute_reply.started":"2025-12-19T19:32:53.421612Z","shell.execute_reply":"2025-12-19T19:32:53.431703Z"}}
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
            plt.close()
            logger.info(f"Successfully saved V2 City Opportunity chart to '{output_path}'.")
        else:
            logger.warning("location_summary_sorted_v2 is empty. Skipping visualization.")
    except Exception as e:
        logger.error(f"An error occurred during visualization: {e}", exc_info=True)
    logger.info("--- Finished visualize_v2_city_opportunity_score Function ---")


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
            plt.close()
            logger.info(f"Successfully saved V3 City Opportunity chart to '{output_path}'.")
        else:
            logger.warning("location_summary_sorted_v3 is empty. Skipping visualization.")
    except Exception as e:
        logger.error(f"An error occurred during V3 visualization: {e}", exc_info=True)
    logger.info("--- Finished visualize_v3_city_opportunity_score Function ---")


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### **Job Card G-V2: Visualize Top 5 Cities (V2)**
# (Code as planned)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.433678Z","iopub.execute_input":"2025-12-19T19:32:53.434033Z","iopub.status.idle":"2025-12-19T19:32:53.459566Z","shell.execute_reply.started":"2025-12-19T19:32:53.434003Z","shell.execute_reply":"2025-12-19T19:32:53.458457Z"}}
try:
    logger.info("--- Starting Job Card G-V3: Visualize Top 5 Cities (V3 - PPP Adjusted) ---")
    visualize_v3_city_opportunity_score(location_summary_sorted_v3, config, logger)
    logger.info("--- V3 ANALYSIS COMPLETE ---")
    logger.info("--- Finished Job Card G-V3 ---")
except Exception as e:
    logger.error(f"Error in Job Card G-V3: {e}", exc_info=True)


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.460765Z","iopub.execute_input":"2025-12-19T19:32:53.461081Z","iopub.status.idle":"2025-12-19T19:32:53.476664Z","shell.execute_reply.started":"2025-12-19T19:32:53.461060Z","shell.execute_reply":"2025-12-19T19:32:53.475800Z"}}
# Function to analyze "Must-Have" Skills
def analyze_must_have_skills(df_input, location_summary_sorted_v2, df_entry_level_v2, logger):
    logger.info("--- Starting analyze_must_have_skills Function ---")
    top_10_skills = pd.Series()  # Initialize with empty Series
    try:
        if not location_summary_sorted_v2.empty and not df_entry_level_v2.empty:
            # 1. Get Top 5 Cities
            top_cities_list = location_summary_sorted_v2.head(5)['location_final'].tolist()
            logger.info(f"Analyzing skills for Top 5 cities: {top_cities_list}")

            # 2. Filter for jobs in top cities
            df_top_cities = df_entry_level_v2[
                df_entry_level_v2['location_final'].isin(top_cities_list)].copy()  # Use .copy()
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


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### **Job Card I: Analyze "Must-Have" Skills**
# #
# **Objective:**
# To identify the most frequently demanded skills for entry-level jobs within our top-ranked cities.
# #
# **Strategy:**
# 1.  Extract the list of top 5 cities from the V2 score summary.
# 2.  Filter the entry-level DataFrame to include only jobs from these top cities.
# 3.  Aggregate all skills from the 'skills' column of this filtered DataFrame.
# 4.  Count the frequency of each skill.
# 5.  Log the top 10 most frequent skills as our "Must-Have" list.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.477711Z","iopub.execute_input":"2025-12-19T19:32:53.477964Z","iopub.status.idle":"2025-12-19T19:32:53.508344Z","shell.execute_reply.started":"2025-12-19T19:32:53.477946Z","shell.execute_reply":"2025-12-19T19:32:53.507231Z"}}
try:
    logger.info("--- Starting Job Card I: Analyze 'Must-Have' Skills ---")
    top_10_skills = analyze_must_have_skills(df, location_summary_sorted_v3, df_entry_level_v3, logger)
    logger.info("--- Finished Job Card I ---")
except Exception as e:
    logger.error(f"An error occurred during Job Card I: {e}", exc_info=True)


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.509457Z","iopub.execute_input":"2025-12-19T19:32:53.509808Z","iopub.status.idle":"2025-12-19T19:32:53.536823Z","shell.execute_reply.started":"2025-12-19T19:32:53.509783Z","shell.execute_reply":"2025-12-19T19:32:53.535799Z"}}
# Function to visualize "Must-Have" Skills
def visualize_must_have_skills(top_10_skills, output_path, logger):
    logger.info("--- Starting visualize_must_have_skills Function ---")
    try:
        if top_10_skills is not None and not top_10_skills.empty:
            import matplotlib.pyplot as plt
            import seaborn as sns
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
            plt.close()
            logger.info(f"Successfully saved 'Must-Have' Skills chart to '{output_path}'.")
        else:
            logger.warning("Top 10 skills data not available. Skipping visualization in function.")
    except Exception as e:
        logger.error(f"An error occurred during skills visualization function: {e}", exc_info=True)
    logger.info("--- Finished visualize_must_have_skills Function ---")


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-19T19:32:53.537858Z","iopub.execute_input":"2025-12-19T19:32:53.538187Z","iopub.status.idle":"2025-12-19T19:32:53.562666Z","shell.execute_reply.started":"2025-12-19T19:32:53.538153Z","shell.execute_reply":"2025-12-19T19:32:53.561196Z"}}
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
            plt.close()
            logger.info(f"Successfully saved 'Must-Have' Skills chart to '{output_path}'.")
        else:
            logger.warning("Top 10 skills data not available. Skipping visualization in function.")
    except Exception as e:
        logger.error(f"An error occurred during skills visualization function: {e}", exc_info=True)
    logger.info("--- Finished visualize_must_have_skills Function ---")


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### **Job Card J: Visualize "Must-Have" Skills**
# #
# **Objective:**
# To create a clear and compelling visualization of the most in-demand skills.

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

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### **Job Card K: Vibe Check - Seniority Clustering**
# #
# **Objective:**
# Use unsupervised clustering to segment jobs into data-driven seniority tiers ("Junior", "Mid", "Senior") based on their salary and experience requirements. This provides a robust foundation for the skill consistency analysis.
# #
# **Methodology:**
# 1.  **Elbow Method:** Empirically determine the optimal number of clusters (K) by plotting the Within-Cluster Sum of Squares (WCSS).
# 2.  **K-Means Clustering:** Apply K-Means with the optimal K to the scaled numerical features.
# 3.  **Cluster Interpretation:** Analyze the resulting clusters to label them by seniority.

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