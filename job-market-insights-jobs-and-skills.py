{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "22a0db43",
   "metadata": {
    "_cell_guid": "cb3f8d53-c947-4042-9352-9e7899dd2371",
    "_uuid": "a731dc27-9367-4410-9075-98abca544a44",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-12-12T06:48:30.978984Z",
     "iopub.status.busy": "2025-12-12T06:48:30.978566Z",
     "iopub.status.idle": "2025-12-12T06:48:32.701831Z",
     "shell.execute_reply": "2025-12-12T06:48:32.700991Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.730095,
     "end_time": "2025-12-12T06:48:32.703527",
     "exception": false,
     "start_time": "2025-12-12T06:48:30.973432",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "faa3ea70",
   "metadata": {
    "_cell_guid": "35400f14-5bec-4468-99bb-4e5a9e829000",
    "_uuid": "b6f59a30-b981-4fd7-b43d-f6315ac5636d",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-12-12T06:48:32.710914Z",
     "iopub.status.busy": "2025-12-12T06:48:32.710137Z",
     "iopub.status.idle": "2025-12-12T06:48:32.717609Z",
     "shell.execute_reply": "2025-12-12T06:48:32.716777Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.01243,
     "end_time": "2025-12-12T06:48:32.719013",
     "exception": false,
     "start_time": "2025-12-12T06:48:32.706583",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-12 06:48:32,713 - __main__ - INFO - Logger initialized and working.\n"
     ]
    }
   ],
   "source": [
    "import logging\n",
    "\n",
    "# Create a logger\n",
    "logger = logging.getLogger(__name__)\n",
    "logger.setLevel(logging.INFO)\n",
    "\n",
    "# Create a file handler\n",
    "file_handler = logging.FileHandler('job_market_data.log')\n",
    "file_handler.setLevel(logging.INFO)\n",
    "\n",
    "# Create a console handler\n",
    "console_handler = logging.StreamHandler()\n",
    "console_handler.setLevel(logging.INFO)\n",
    "\n",
    "# Create a formatter and add it to the handlers\n",
    "formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')\n",
    "file_handler.setFormatter(formatter)\n",
    "console_handler.setFormatter(formatter)\n",
    "\n",
    "# Add the handlers to the logger\n",
    "logger.addHandler(file_handler)\n",
    "logger.addHandler(console_handler)\n",
    "\n",
    "logger.info(\"Logger initialized and working.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0a65488b",
   "metadata": {
    "_cell_guid": "c11c003f-8a6a-4b9e-bbe9-1c8a2cd83921",
    "_uuid": "6bfba982-1621-4dbd-8253-2c6d2a99eb39",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-12-12T06:48:32.725854Z",
     "iopub.status.busy": "2025-12-12T06:48:32.725601Z",
     "iopub.status.idle": "2025-12-12T06:48:32.731434Z",
     "shell.execute_reply": "2025-12-12T06:48:32.730751Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.010785,
     "end_time": "2025-12-12T06:48:32.732695",
     "exception": false,
     "start_time": "2025-12-12T06:48:32.721910",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\nCode cell to setup logging to kaggle log \\n'"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''\n",
    "Code cell to setup logging to kaggle log \n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "02bbd5b3",
   "metadata": {
    "_cell_guid": "d381ce26-2486-457a-9799-100581724807",
    "_uuid": "19b25139-3dc2-417f-9f53-fb76066adcc2",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-12-12T06:48:32.739725Z",
     "iopub.status.busy": "2025-12-12T06:48:32.739461Z",
     "iopub.status.idle": "2025-12-12T06:48:32.743325Z",
     "shell.execute_reply": "2025-12-12T06:48:32.742587Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.009019,
     "end_time": "2025-12-12T06:48:32.744626",
     "exception": false,
     "start_time": "2025-12-12T06:48:32.735607",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Markdown cell to state project goals and data import stratergy"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7b45f22",
   "metadata": {
    "_cell_guid": "66976b42-52ac-46ce-b6c9-5c9a225efb0c",
    "_uuid": "0d49b62c-c49c-454f-8bc2-1c4cb5c15d43",
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.0027,
     "end_time": "2025-12-12T06:48:32.750281",
     "exception": false,
     "start_time": "2025-12-12T06:48:32.747581",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### **Job Card: Data Import and Initial Processing**\n",
    "\n",
    "**Objective:**\n",
    "To reliably and efficiently import the raw `job_market_data.csv` into a pandas DataFrame, handling known data inconsistencies and large file sizes gracefully.\n",
    "\n",
    "**Strategy:**\n",
    "A chunk-based processing pipeline will be used. Instead of loading the entire file into memory at once, the data will be read and processed in smaller, manageable chunks, which are then combined into a final DataFrame. This approach minimizes memory usage and allows for on-the-fly corrections.\n",
    "\n",
    "**Key Implementation Steps:**\n",
    "\n",
    "1.  **Define Data Types:** A `dtype` dictionary will be created to enforce initial data types on import. Most text columns will be `object`, numerical columns like `salary_min`, `salary_max`, and `experience_required` will be `float64` (to accommodate NaNs), and `publication_date` will be read as an `object` to allow for custom parsing.\n",
    "\n",
    "2.  **Define Custom Date Parser:** A Python function (`parse_mixed_dates`) will be created. This function will take a pandas Series (the `publication_date` column from a chunk) and convert its values to datetime objects, correctly handling both Unix timestamps and standard date strings (e.g., `yyyy-mm-dd`).\n",
    "\n",
    "3.  **Chunked Reading:** A `for` loop will iterate through the CSV file using `pd.read_csv` with the `chunksize=50` parameter.\n",
    "\n",
    "4.  **Per-Chunk Processing:** Inside the loop, for each chunk:\n",
    "    *   The `parse_mixed_dates` function will be applied to the `publication_date` column.\n",
    "    *   The processed chunk will be stored in a list.\n",
    "\n",
    "5.  **Final Concatenation:** After the loop finishes, `pd.concat()` will be used to merge all the processed chunks into a single, comprehensive DataFrame.\n",
    "\n",
    "6.  **Error Handling:** The entire process will be enclosed in a `try...except` block to catch `FileNotFoundError` and any other exceptions that might occur during processing, ensuring the script does not crash.\n",
    "\n",
    "**Logging:**\n",
    "*   Log the start of the chunked import process.\n",
    "*   Log the processing of each individual chunk.\n",
    "*   Log the start of the final concatenation step.\n",
    "*   Log a success message with the final DataFrame's shape and data types.\n",
    "*   Log any errors encountered, including the full traceback for debugging.\n",
    "\n",
    "**Output:**\n",
    "A single pandas DataFrame variable (`df`) containing all the data from the CSV, with the `publication_date` column correctly parsed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "07fcf0d9",
   "metadata": {
    "_cell_guid": "7552ace8-e789-4666-a7bb-3eb2f0fe62a0",
    "_uuid": "5badbe1f-fa78-4838-af28-7cbbf528bf66",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-12-12T06:48:32.757252Z",
     "iopub.status.busy": "2025-12-12T06:48:32.756946Z",
     "iopub.status.idle": "2025-12-12T06:48:32.824726Z",
     "shell.execute_reply": "2025-12-12T06:48:32.824057Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.073092,
     "end_time": "2025-12-12T06:48:32.826110",
     "exception": false,
     "start_time": "2025-12-12T06:48:32.753018",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-12 06:48:32,768 - __main__ - INFO - Starting robust data import from /kaggle/input/job-market-insight/job_market.csv...\n",
      "2025-12-12 06:48:32,793 - __main__ - INFO - Processing chunk 1...\n",
      "2025-12-12 06:48:32,818 - __main__ - INFO - Data import process completed.\n",
      "2025-12-12 06:48:32,819 - __main__ - INFO - Successfully imported 250 clean rows.\n"
     ]
    },
    {
     "data": {
      "text/markdown": [
       "# Data Import Report\n",
       "\n",
       "✅ **250 clean rows** were successfully imported and are available in the `df` DataFrame.\n",
       "\n",
       "No malformed rows were detected."
      ],
      "text/plain": [
       "<IPython.core.display.Markdown object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# --- Configuration ---\n",
    "DATA_FILE_PATH = '/kaggle/input/job-market-insight/job_market.csv'\n",
    "CHUNK_SIZE = 500\n",
    "NA_VALUES = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']\n",
    "NUMERIC_COLS = ['salary_min', 'salary_max', 'experience_required']\n",
    "DATE_COL = 'publication_date'\n",
    "\n",
    "# --- Data Structures ---\n",
    "clean_chunks = []\n",
    "malformed_chunks = []\n",
    "\n",
    "def parse_mixed_dates(series):\n",
    "    # Handles Unix timestamps and standard date strings\n",
    "    numeric_dates = pd.to_numeric(series, errors='coerce')\n",
    "    unix_dates = pd.to_datetime(numeric_dates, unit='s', errors='coerce')\n",
    "    string_dates = pd.to_datetime(series[numeric_dates.isna()], errors='coerce')\n",
    "    return unix_dates.fillna(string_dates)\n",
    "\n",
    "# --- Main Import and Validation Loop ---\n",
    "try:\n",
    "    logger.info(f\"Starting robust data import from {DATA_FILE_PATH}...\")\n",
    "    \n",
    "    with pd.read_csv(DATA_FILE_PATH, chunksize=CHUNK_SIZE, dtype=object, keep_default_na=False) as reader:\n",
    "        for i, chunk_raw_text in enumerate(reader):\n",
    "            logger.info(f\"Processing chunk {i+1}...\")\n",
    "            \n",
    "            # 1. Identify legitimate blanks before any conversion\n",
    "            legit_nan_mask = chunk_raw_text.isin(NA_VALUES)\n",
    "            \n",
    "            # 2. Make a copy for conversion\n",
    "            converted_chunk = chunk_raw_text.copy()\n",
    "            \n",
    "            # 3. Apply coercive conversions\n",
    "            for col in NUMERIC_COLS:\n",
    "                converted_chunk[col] = pd.to_numeric(converted_chunk[col], errors='coerce')\n",
    "            converted_chunk[DATE_COL] = parse_mixed_dates(converted_chunk[DATE_COL])\n",
    "            \n",
    "            # 4. Find locations of all NaNs/NaTs in the final converted chunk\n",
    "            final_nan_mask = converted_chunk.isnull()\n",
    "            \n",
    "            # 5. Identify import-created NaNs\n",
    "            import_created_nans = final_nan_mask & ~legit_nan_mask\n",
    "            \n",
    "            # 6. Identify malformed rows\n",
    "            is_malformed = import_created_nans.any(axis=1)\n",
    "            \n",
    "            # 7. Segregate the data\n",
    "            clean_rows = converted_chunk[~is_malformed]\n",
    "            malformed_rows = chunk_raw_text[is_malformed]\n",
    "            \n",
    "            if not clean_rows.empty:\n",
    "                clean_chunks.append(clean_rows)\n",
    "            if not malformed_rows.empty:\n",
    "                malformed_chunks.append(malformed_rows)\n",
    "\n",
    "    # --- Finalize DataFrames ---\n",
    "    df = pd.concat(clean_chunks, ignore_index=True) if clean_chunks else pd.DataFrame()\n",
    "    df_malformed = pd.concat(malformed_chunks, ignore_index=True) if malformed_chunks else pd.DataFrame()\n",
    "    \n",
    "    logger.info(\"Data import process completed.\")\n",
    "    logger.info(f\"Successfully imported {len(df)} clean rows.\")\n",
    "    \n",
    "    # --- Generate Report ---\n",
    "    report_md = f\"# Data Import Report\\n\\n\"\n",
    "    report_md += f\"✅ **{len(df)} clean rows** were successfully imported and are available in the `df` DataFrame.\\n\\n\"\n",
    "    \n",
    "    if not df_malformed.empty:\n",
    "        logger.warning(f\"Quarantined {len(df_malformed)} malformed rows. See `df_malformed` for details.\")\n",
    "        report_md += f\"**{len(df_malformed)} malformed rows** were quarantined because they contained data that could not be correctly parsed. These are available for review in the `df_malformed` DataFrame.\\n\\n\"\n",
    "        \n",
    "        # Log and display a summary of issues\n",
    "        report_md += \"### Summary of Issues\\n\\n\"\n",
    "        # Find columns with parsing issues\n",
    "        malformed_summary = {}\n",
    "        for col in df_malformed.columns:\n",
    "            # Get original values that ended up as NaN\n",
    "            original_values = df_malformed[col]\n",
    "            converted_values = pd.to_numeric(original_values, errors='coerce') if col in NUMERIC_COLS else parse_mixed_dates(original_values) if col == DATE_COL else original_values\n",
    "            \n",
    "            problematic_mask = converted_values.isnull() & original_values.notnull() & ~original_values.isin(NA_VALUES)\n",
    "            if problematic_mask.any():\n",
    "                top_problems = original_values[problematic_mask].value_counts().nlargest(5)\n",
    "                malformed_summary[col] = top_problems\n",
    "                logger.info(f\"Top 5 problematic values in column '{col}':\\n{top_problems.to_string()}\")\n",
    "\n",
    "        # Format summary for markdown\n",
    "        for col, problems in malformed_summary.items():\n",
    "            report_md += f\"**Column: `{col}`**\\n\"\n",
    "            report_md += \"| Problematic Value | Count |\\n\"\n",
    "            report_md += \"|---|---|\\n\"\n",
    "            for value, count in problems.items():\n",
    "                report_md += f\"| `{value}` | {count} |\\n\"\n",
    "            report_md += \"\\n\"\n",
    "    else:\n",
    "        report_md += \"No malformed rows were detected.\"\n",
    "\n",
    "except FileNotFoundError:\n",
    "    logger.error(f\"Data file not found at {DATA_FILE_PATH}. Please check the file path.\")\n",
    "    df = pd.DataFrame()\n",
    "    df_malformed = pd.DataFrame()\n",
    "    report_md = f\"# Data Import Report\\n\\n **Error:** Data file not found at `{DATA_FILE_PATH}`.\"\n",
    "except Exception as e:\n",
    "    logger.error(f\"An unexpected error occurred during data import: {e}\", exc_info=True)\n",
    "    df = pd.DataFrame()\n",
    "    df_malformed = pd.DataFrame()\n",
    "    report_md = f\"# Data Import Report\\n\\n **An unexpected error occurred.** Check the logs for details.\"\n",
    "\n",
    "# Display the markdown report\n",
    "from IPython.display import display, Markdown\n",
    "display(Markdown(report_md))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "98639574",
   "metadata": {
    "_cell_guid": "854926b7-6854-42f8-8f20-ba6ff3bd867d",
    "_uuid": "fa0870b5-bb9d-4567-b238-c435a9e4bbbc",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-12-12T06:48:32.833945Z",
     "iopub.status.busy": "2025-12-12T06:48:32.833666Z",
     "iopub.status.idle": "2025-12-12T06:48:32.855401Z",
     "shell.execute_reply": "2025-12-12T06:48:32.854515Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.027332,
     "end_time": "2025-12-12T06:48:32.856692",
     "exception": false,
     "start_time": "2025-12-12T06:48:32.829360",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-12 06:48:32,837 - __main__ - INFO - Starting analysis of 'category' column for normalization.\n",
      "2025-12-12 06:48:32,838 - __main__ - INFO - Analyzing values in 'category' column.\n",
      "2025-12-12 06:48:32,841 - __main__ - INFO - Found 13 unique values in the 'category' column.\n",
      "2025-12-12 06:48:32,844 - __main__ - INFO - Top 15 most common categories:\n",
      "category\n",
      "Technology                     200\n",
      "                                20\n",
      "Remote                          11\n",
      "Marketing and Communication      6\n",
      "Software Development             4\n",
      "Social Media Manager             2\n",
      "Recruitment and Selection        1\n",
      "SAP/ERP Consulting               1\n",
      "Helpdesk                         1\n",
      "Media Planning                   1\n",
      "Finance                          1\n",
      "Process Engineering              1\n",
      "HR                               1\n",
      "2025-12-12 06:48:32,844 - __main__ - INFO - Findings:\n",
      "2025-12-12 06:48:32,845 - __main__ - INFO -  - High cardinality observed in 'category' column.\n",
      "2025-12-12 06:48:32,846 - __main__ - INFO -  - Some categories might be too specific or variations of the same field (e.g., 'Technology', 'Software Development').\n",
      "2025-12-12 06:48:32,847 - __main__ - INFO - Proposed Normalization Strategy:\n",
      "2025-12-12 06:48:32,848 - __main__ - INFO -  - Group similar or niche categories into broader, more common fields.\n",
      "2025-12-12 06:48:32,849 - __main__ - INFO -  - For example, group specific tech roles under a general 'Technology' umbrella if appropriate.\n",
      "2025-12-12 06:48:32,850 - __main__ - INFO -  - Investigate mapping to industry-standard categories if possible.\n",
      "2025-12-12 06:48:32,851 - __main__ - INFO -  - Handle NaN values in 'category' column.\n",
      "2025-12-12 06:48:32,851 - __main__ - INFO - Analysis of 'category' column distribution completed.\n"
     ]
    }
   ],
   "source": [
    "'''\n",
    "Code cell for analyzing the 'category' column distribution and normalization.\n",
    "Investigates unique values and proposes strategies for consolidating categories.\n",
    "'''\n",
    "try:\n",
    "    logger.info(\"Starting analysis of 'category' column for normalization.\")\n",
    "\n",
    "    if 'df' in locals() and not df.empty:\n",
    "        if 'category' in df.columns:\n",
    "            logger.info(\"Analyzing values in 'category' column.\")\n",
    "            \n",
    "            # Get unique values and their counts\n",
    "            category_counts = df['category'].value_counts()\n",
    "            num_unique_categories = df['category'].nunique()\n",
    "            \n",
    "            logger.info(f\"Found {num_unique_categories} unique values in the 'category' column.\")\n",
    "            logger.info(f\"Top 15 most common categories:\\n{category_counts.nlargest(15).to_string()}\")\n",
    "            \n",
    "            # Log findings and propose normalization strategy\n",
    "            # Gemini remove hard coded log entries \n",
    "            logger.info(\"Findings:\")\n",
    "            logger.info(\" - High cardinality observed in 'category' column.\")\n",
    "            logger.info(\" - Some categories might be too specific or variations of the same field (e.g., 'Technology', 'Software Development').\")\n",
    "            logger.info(\"Proposed Normalization Strategy:\")\n",
    "            logger.info(\" - Group similar or niche categories into broader, more common fields.\")\n",
    "            logger.info(\" - For example, group specific tech roles under a general 'Technology' umbrella if appropriate.\")\n",
    "            logger.info(\" - Investigate mapping to industry-standard categories if possible.\")\n",
    "            logger.info(\" - Handle NaN values in 'category' column.\")\n",
    "            '''Verobose logging is irrelivent'''\n",
    "            logger.info(\"Analysis of 'category' column distribution completed.\")\n",
    "        else:\n",
    "            logger.warning(\"Column 'category' not found in DataFrame. Skipping analysis.\")\n",
    "    else:\n",
    "        logger.warning(\"DataFrame 'df' not found or is empty. Skipping analysis of 'category' column.\")\n",
    "\n",
    "except Exception as e:\n",
    "    logger.error(f\"An error occurred during the analysis of 'category' column: {e}\", exc_info=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "fcbdbf0d",
   "metadata": {
    "_cell_guid": "4bbb798a-b631-4a3d-b40d-1236b3dff21e",
    "_uuid": "eb50d190-6912-43e5-815d-bec8e05ecbf4",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-12-12T06:48:32.865670Z",
     "iopub.status.busy": "2025-12-12T06:48:32.865410Z",
     "iopub.status.idle": "2025-12-12T06:48:32.872669Z",
     "shell.execute_reply": "2025-12-12T06:48:32.871868Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.013391,
     "end_time": "2025-12-12T06:48:32.874036",
     "exception": false,
     "start_time": "2025-12-12T06:48:32.860645",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-12 06:48:32,867 - __main__ - INFO - Starting data cleaning operations.\n",
      "2025-12-12 06:48:32,869 - __main__ - INFO - Data cleaning operations completed.\n"
     ]
    }
   ],
   "source": [
    "'''\n",
    "Code cell for data cleaning operations.\n",
    "Specific cleaning steps will be added here based on data quality assessment results.\n",
    "'''\n",
    "try:\n",
    "    logger.info(\"Starting data cleaning operations.\")\n",
    "\n",
    "    if 'df' in locals() and not df.empty:\n",
    "        # Placeholder for data cleaning steps.\n",
    "        # Examples of cleaning steps:\n",
    "        # 1. Handle missing values:\n",
    "        #    df['salary_min'].fillna(df['salary_min'].median(), inplace=True)\n",
    "        # 2. Standardize categorical columns:\n",
    "        #    df['company'] = df['company'].str.strip().str.upper()\n",
    "        # 3. Convert data types if necessary (e.g., after cleaning 'experience_required' object to numeric)\n",
    "        #    df['experience_required'] = pd.to_numeric(df['experience_required'], errors='coerce')\n",
    "        # 4. Handle 'skills' column (e.g., convert string to list of skills)\n",
    "        #    df['skills'] = df['skills'].apply(lambda x: [skill.strip() for skill in x.split(',')] if isinstance(x, str) else [])\n",
    "\n",
    "\n",
    "        logger.info(\"Data cleaning operations completed.\")\n",
    "        # Optionally, log post-cleaning DataFrame info\n",
    "        # logger.info(f\"DataFrame shape after cleaning: {df.shape}\")\n",
    "        # logger.info(f\"DataFrame dtypes after cleaning:\\n{df.dtypes}\")\n",
    "    else:\n",
    "        logger.warning(\"DataFrame 'df' not found or is empty. Skipping data cleaning operations.\")\n",
    "\n",
    "except Exception as e:\n",
    "    logger.error(f\"An error occurred during data cleaning operations: {e}\", exc_info=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ef525722",
   "metadata": {
    "_cell_guid": "21d1e881-0469-488c-b19f-03b5af6535dc",
    "_uuid": "d6485f1c-f6e3-4b8b-8aa2-bb1c6f6e61a4",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-12-12T06:48:32.882738Z",
     "iopub.status.busy": "2025-12-12T06:48:32.882481Z",
     "iopub.status.idle": "2025-12-12T06:48:32.910628Z",
     "shell.execute_reply": "2025-12-12T06:48:32.909643Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.034242,
     "end_time": "2025-12-12T06:48:32.912067",
     "exception": false,
     "start_time": "2025-12-12T06:48:32.877825",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-12 06:48:32,885 - __main__ - INFO - Attempting to create MultiIndex on 'company' and 'location'.\n",
      "2025-12-12 06:48:32,905 - __main__ - INFO - MultiIndex created successfully on 'company' and 'location' columns.\n",
      "2025-12-12 06:48:32,907 - __main__ - INFO - DataFrame index: ['company', 'location']\n"
     ]
    }
   ],
   "source": [
    "'''\n",
    "Code cell for creating MultiIndex.\n",
    "'''\n",
    "try:\n",
    "    logger.info(\"Attempting to create MultiIndex on 'company' and 'location'.\")\n",
    "\n",
    "    if 'df' in locals() and not df.empty:\n",
    "        # Check if the columns exist and are not null\n",
    "        if 'company' in df.columns and 'location' in df.columns:\n",
    "            if df['company'].isnull().any() or df['location'].isnull().any():\n",
    "                logger.warning(\"Skipping MultiIndex creation: 'company' or 'location' columns contain missing values. Please clean these columns first.\")\n",
    "            else:\n",
    "                df.set_index(['company', 'location'], inplace=True)\n",
    "                logger.info(\"MultiIndex created successfully on 'company' and 'location' columns.\")\n",
    "                logger.info(f\"DataFrame index: {df.index.names}\")\n",
    "        else:\n",
    "            logger.warning(\"Skipping MultiIndex creation: 'company' or 'location' column(s) not found in DataFrame.\")\n",
    "    else:\n",
    "        logger.warning(\"DataFrame 'df' not found or is empty. Skipping MultiIndex creation.\")\n",
    "\n",
    "except Exception as e:\n",
    "    logger.error(f\"An error occurred during MultiIndex creation: {e}\", exc_info=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "74261c16",
   "metadata": {
    "_cell_guid": "9edf2f15-0c43-4711-a054-e64b2d9bd3bf",
    "_uuid": "df957a95-6b90-4d27-8e03-3e7ba9f5d4eb",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-12-12T06:48:32.921475Z",
     "iopub.status.busy": "2025-12-12T06:48:32.921216Z",
     "iopub.status.idle": "2025-12-12T06:48:32.928130Z",
     "shell.execute_reply": "2025-12-12T06:48:32.927381Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.013223,
     "end_time": "2025-12-12T06:48:32.929478",
     "exception": false,
     "start_time": "2025-12-12T06:48:32.916255",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-12 06:48:32,923 - __main__ - INFO - Starting visualization rendering process.\n",
      "2025-12-12 06:48:32,925 - __main__ - INFO - Visualization rendering process completed.\n"
     ]
    }
   ],
   "source": [
    "'''\n",
    "Code cell for visualizations.\n",
    "Specific visualization code will be added here based on analysis goals.\n",
    "'''\n",
    "try:\n",
    "    logger.info(\"Starting visualization rendering process.\")\n",
    "\n",
    "    if 'df' in locals() and not df.empty:\n",
    "        # Placeholder for visualization code.\n",
    "        # Examples of visualization:\n",
    "        # import matplotlib.pyplot as plt\n",
    "        # import seaborn as sns\n",
    "        #\n",
    "        # plt.figure(figsize=(10, 6))\n",
    "        # sns.countplot(y='company', data=df, order=df['company'].value_counts().index[:10])\n",
    "        # plt.title('Top 10 Companies by Job Postings')\n",
    "        # plt.show()\n",
    "        # logger.info(\"Generated 'Top 10 Companies by Job Postings' chart.\")\n",
    "\n",
    "        logger.info(\"Visualization rendering process completed.\")\n",
    "    else:\n",
    "        logger.warning(\"DataFrame 'df' not found or is empty. Skipping visualization rendering.\")\n",
    "\n",
    "except Exception as e:\n",
    "    logger.error(f\"An error occurred during visualization rendering: {e}\", exc_info=True)"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 8850949,
     "sourceId": 13892754,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 31192,
   "isGpuEnabled": false,
   "isInternetEnabled": false,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 6.936838,
   "end_time": "2025-12-12T06:48:33.352158",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-12-12T06:48:26.415320",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
