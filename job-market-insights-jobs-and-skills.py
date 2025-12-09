# %% [code] {"execution":{"iopub.status.busy":"2025-12-08T07:52:23.086242Z","iopub.execute_input":"2025-12-08T07:52:23.086597Z","iopub.status.idle":"2025-12-08T07:52:25.679644Z","shell.execute_reply.started":"2025-12-08T07:52:23.086543Z","shell.execute_reply":"2025-12-08T07:52:25.678364Z"},"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"jupyter":{"outputs_hidden":false}}
import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler('job_market_data.log')
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

# %% [code]
'''
Code cell to setup logging to kaggle log 
'''


# %% [code] {"jupyter":{"outputs_hidden":false}}
# Markdown cell to state project goals and data import stratergy

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### **Job Card: Data Import and Initial Processing**
# 
# **Objective:**
# To reliably and efficiently import the raw `job_market_data.csv` into a pandas DataFrame, handling known data inconsistencies and large file sizes gracefully.
# 
# **Strategy:**
# A chunk-based processing pipeline will be used. Instead of loading the entire file into memory at once, the data will be read and processed in smaller, manageable chunks, which are then combined into a final DataFrame. This approach minimizes memory usage and allows for on-the-fly corrections.
# 
# **Key Implementation Steps:**
# 
# 1.  **Define Data Types:** A `dtype` dictionary will be created to enforce initial data types on import. Most text columns will be `object`, numerical columns like `salary_min`, `salary_max`, and `experience_required` will be `float64` (to accommodate NaNs), and `publication_date` will be read as an `object` to allow for custom parsing.
# 
# 2.  **Define Custom Date Parser:** A Python function (`parse_mixed_dates`) will be created. This function will take a pandas Series (the `publication_date` column from a chunk) and convert its values to datetime objects, correctly handling both Unix timestamps and standard date strings (e.g., `yyyy-mm-dd`).
# 
# 3.  **Chunked Reading:** A `for` loop will iterate through the CSV file using `pd.read_csv` with the `chunksize=50` parameter.
# 
# 4.  **Per-Chunk Processing:** Inside the loop, for each chunk:
#     *   The `parse_mixed_dates` function will be applied to the `publication_date` column.
#     *   The processed chunk will be stored in a list.
# 
# 5.  **Final Concatenation:** After the loop finishes, `pd.concat()` will be used to merge all the processed chunks into a single, comprehensive DataFrame.
# 
# 6.  **Error Handling:** The entire process will be enclosed in a `try...except` block to catch `FileNotFoundError` and any other exceptions that might occur during processing, ensuring the script does not crash.
# 
# **Logging:**
# *   Log the start of the chunked import process.
# *   Log the processing of each individual chunk.
# *   Log the start of the final concatenation step.
# *   Log a success message with the final DataFrame's shape and data types.
# *   Log any errors encountered, including the full traceback for debugging.
# 
# **Output:**
# A single pandas DataFrame variable (`df`) containing all the data from the CSV, with the `publication_date` column correctly parsed.

# %% [code] {"execution":{"iopub.status.busy":"2025-12-08T08:06:35.709767Z","iopub.execute_input":"2025-12-08T08:06:35.710961Z","iopub.status.idle":"2025-12-08T08:06:35.717601Z","shell.execute_reply.started":"2025-12-08T08:06:35.710922Z","shell.execute_reply":"2025-12-08T08:06:35.716420Z"},"jupyter":{"outputs_hidden":false}}
'''
Code cell to import data from CSV file to panda dataframe 
NOTE: date cell contains date formats yyyy-mm-dd and what apprears to be unix date format
nand data 

'''
def parse_mixed_dates(series):
    # Attempt to convert to numeric for Unix timestamps, coercing errors to NaN
    numeric_dates = pd.to_numeric(series, errors='coerce')
    # Convert Unix timestamps (assuming they are in seconds)
    unix_dates = pd.to_datetime(numeric_dates, unit='s', errors='coerce')
    # For the rest, convert directly, coercing errors to NaT
    string_dates = pd.to_datetime(series[numeric_dates.isna()], errors='coerce')
    # Combine the results
    return unix_dates.fillna(string_dates)

data_types = {
    'job_title': 'object',
    'company': 'object',
    'location': 'object',
    'job_type': 'object',
    'category': 'object',
    'salary_min': 'float64',
    'salary_max': 'float64',
    'experience_required': 'float64',
    'skills': 'object',
    'publication_date': 'object' # Read as object to handle mixed formats
}

processed_chunks = []
DATA_FILE_PATH = '/kaggle/input/job-market-insight/job_market.csv'
CHUNK_SIZE = 50

try:
    logger.info(f"Starting chunked data import from {DATA_FILE_PATH} with chunk size {CHUNK_SIZE}.")
    
    with pd.read_csv(DATA_FILE_PATH, dtype=data_types, chunksize=CHUNK_SIZE) as reader:
        for i, chunk in enumerate(reader):
            logger.info(f"Processing chunk {i+1}...")
            
            # Custom date parsing
            chunk['publication_date'] = parse_mixed_dates(chunk['publication_date'])
            
            # Further per-chunk processing can be added here
            
            processed_chunks.append(chunk)

    logger.info("All chunks processed. Concatenating into a single DataFrame.")
    df = pd.concat(processed_chunks, ignore_index=True)
    
    logger.info("Data imported and concatenated successfully.")
    logger.info(f"Final DataFrame shape: {df.shape}")
    logger.info(f"Final DataFrame columns and their dtypes:\n{df.dtypes}")

except FileNotFoundError:
    logger.error(f"Data file not found at {DATA_FILE_PATH}. Please check the file path.")
    df = pd.DataFrame() # Ensure df exists even if import fails
except Exception as e:
    logger.error(f"An error occurred during chunked data import: {e}", exc_info=True)
    df = pd.DataFrame() # Ensure df exists even if import fails


# %% [code] {"jupyter":{"outputs_hidden":false}}
'''
Code cell for data quality assessment
'''
try:
    logger.info("Starting data quality assessment.")

    # Check if df exists and is not empty
    if 'df' in locals() and not df.empty:
        # 1. Missing Value Analysis (NaNs)
        missing_values_count = df.isnull().sum()
        missing_values_percent = (missing_values_count / len(df)) * 100
        logger.info("--- Missing Value Analysis ---")
        for col, count in missing_values_count.items():
            if count > 0:
                logger.info(f"Column '{col}': {count} missing values ({missing_values_percent[col]:.2f}%)")

        # 2. Row Completeness (Shape)
        complete_rows = df.dropna().shape[0]
        total_rows = len(df)
        completeness_percent = (complete_rows / total_rows) * 100
        logger.info("--- Row Completeness ---")
        logger.info(f"{complete_rows} of {total_rows} rows are complete ({completeness_percent:.2f}%).")
        logger.info(f"DataFrame Shape: {df.shape}")


        # 3. Uniqueness and Value Distribution for Categorical Data
        categorical_cols = ['company', 'location', 'job_type', 'category']
        logger.info("--- Uniqueness & Value Counts for Categorical Columns ---")
        for col in categorical_cols:
            logger.info(f"Number of unique values in '{col}': {df[col].nunique()}")
            logger.info(f"Top 5 most common values for '{col}':")
            logger.info(f"\n{df[col].value_counts().nlargest(5).to_string()}")

        # 4. Descriptive Statistics for Numerical Data
        numerical_cols = ['salary_min', 'salary_max', 'experience_required']
        logger.info("--- Descriptive Statistics for Numerical Columns ---")
        logger.info(f"\n{df[numerical_cols].describe().to_string()}")

        logger.info("Data quality assessment completed.")

    else:
        logger.warning("DataFrame 'df' not found or is empty. Skipping data quality assessment.")

except Exception as e:
    logger.error(f"An error occurred during the investigation of 'experience_required': {e}", exc_info=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
'''
Code cell for analyzing the 'category' column distribution and normalization.
Investigates unique values and proposes strategies for consolidating categories.
'''
try:
    logger.info("Starting analysis of 'category' column for normalization.")

    if 'df' in locals() and not df.empty:
        if 'category' in df.columns:
            logger.info("Analyzing values in 'category' column.")
            
            # Get unique values and their counts
            category_counts = df['category'].value_counts()
            num_unique_categories = df['category'].nunique()
            
            logger.info(f"Found {num_unique_categories} unique values in the 'category' column.")
            logger.info(f"Top 15 most common categories:\n{category_counts.nlargest(15).to_string()}")
            
            # Log findings and propose normalization strategy
            logger.info("Findings:")
            logger.info(" - High cardinality observed in 'category' column.")
            logger.info(" - Some categories might be too specific or variations of the same field (e.g., 'Technology', 'Software Development').")
            logger.info("Proposed Normalization Strategy:")
            logger.info(" - Group similar or niche categories into broader, more common fields.")
            logger.info(" - For example, group specific tech roles under a general 'Technology' umbrella if appropriate.")
            logger.info(" - Investigate mapping to industry-standard categories if possible.")
            logger.info(" - Handle NaN values in 'category' column.")
            
            logger.info("Analysis of 'category' column distribution completed.")
        else:
            logger.warning("Column 'category' not found in DataFrame. Skipping analysis.")
    else:
        logger.warning("DataFrame 'df' not found or is empty. Skipping analysis of 'category' column.")

except Exception as e:
    logger.error(f"An error occurred during the analysis of 'category' column: {e}", exc_info=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
'''
Code cell for data cleaning operations.
Specific cleaning steps will be added here based on data quality assessment results.
'''
try:
    logger.info("Starting data cleaning operations.")

    if 'df' in locals() and not df.empty:
        # Placeholder for data cleaning steps.
        # Examples of cleaning steps:
        # 1. Handle missing values:
        #    df['salary_min'].fillna(df['salary_min'].median(), inplace=True)
        # 2. Standardize categorical columns:
        #    df['company'] = df['company'].str.strip().str.upper()
        # 3. Convert data types if necessary (e.g., after cleaning 'experience_required' object to numeric)
        #    df['experience_required'] = pd.to_numeric(df['experience_required'], errors='coerce')
        # 4. Handle 'skills' column (e.g., convert string to list of skills)
        #    df['skills'] = df['skills'].apply(lambda x: [skill.strip() for skill in x.split(',')] if isinstance(x, str) else [])


        logger.info("Data cleaning operations completed.")
        # Optionally, log post-cleaning DataFrame info
        # logger.info(f"DataFrame shape after cleaning: {df.shape}")
        # logger.info(f"DataFrame dtypes after cleaning:\n{df.dtypes}")
    else:
        logger.warning("DataFrame 'df' not found or is empty. Skipping data cleaning operations.")

except Exception as e:
    logger.error(f"An error occurred during data cleaning operations: {e}", exc_info=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
'''
Code cell for creating MultiIndex.
'''
try:
    logger.info("Attempting to create MultiIndex on 'company' and 'location'.")

    if 'df' in locals() and not df.empty:
        # Check if the columns exist and are not null
        if 'company' in df.columns and 'location' in df.columns:
            if df['company'].isnull().any() or df['location'].isnull().any():
                logger.warning("Skipping MultiIndex creation: 'company' or 'location' columns contain missing values. Please clean these columns first.")
            else:
                df.set_index(['company', 'location'], inplace=True)
                logger.info("MultiIndex created successfully on 'company' and 'location' columns.")
                logger.info(f"DataFrame index: {df.index.names}")
        else:
            logger.warning("Skipping MultiIndex creation: 'company' or 'location' column(s) not found in DataFrame.")
    else:
        logger.warning("DataFrame 'df' not found or is empty. Skipping MultiIndex creation.")

except Exception as e:
    logger.error(f"An error occurred during MultiIndex creation: {e}", exc_info=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
'''
Code cell for visualizations.
Specific visualization code will be added here based on analysis goals.
'''
try:
    logger.info("Starting visualization rendering process.")

    if 'df' in locals() and not df.empty:
        # Placeholder for visualization code.
        # Examples of visualization:
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        #
        # plt.figure(figsize=(10, 6))
        # sns.countplot(y='company', data=df, order=df['company'].value_counts().index[:10])
        # plt.title('Top 10 Companies by Job Postings')
        # plt.show()
        # logger.info("Generated 'Top 10 Companies by Job Postings' chart.")

        logger.info("Visualization rendering process completed.")
    else:
        logger.warning("DataFrame 'df' not found or is empty. Skipping visualization rendering.")

except Exception as e:
    logger.error(f"An error occurred during visualization rendering: {e}", exc_info=True)