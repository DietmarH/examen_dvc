import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import logging


def normalize_and_save_data(base_dir):
    """
    Loads training and testing feature datasets (X_train.csv and X_test.csv) from the processed data directory,
    applies standard normalization (zero mean, unit variance) to all numeric columns using StandardScaler
    (fitted only on X_train), and saves the normalized datasets as new CSV files.

    The function preserves non-numeric columns (if any) without modification and ensures the column order
    remains unchanged. The normalized datasets are saved as 'X_train_scaled.csv' and 'X_test_scaled.csv'
    in the same processed data directory.

        base_dir (str): The base directory of the project. The function expects the processed data to be located at
                        '{base_dir}/data/processed/'.

    Raises:
        FileNotFoundError: If either 'X_train.csv' or 'X_test.csv' is not found in the processed data directory.
        Exception: For any other unexpected errors during data loading or processing.

    Logging:
        The function logs progress and errors using the standard logging module.
    """

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Define the path to the data/processed directory for both input and output
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')
    input_dir = processed_data_dir
    output_dir = processed_data_dir

    logger.info(f"Attempting to load X_train.csv and X_test.csv from: {input_dir}")
    try:
        # Load the training and testing feature sets
        X_train = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))
        logger.info("Data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Input files (X_train.csv or X_test.csv) not found in {input_dir}. Please ensure the 'Data Splitting' script has been run and saved the files correctly.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        return

    logger.info("Initializing StandardScaler and normalizing data...")
    # Select only numeric columns for scaling
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    non_numeric_cols = X_train.columns.difference(numeric_cols)

    scaler = StandardScaler()
    X_train_scaled_numeric = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled_numeric = scaler.transform(X_test[numeric_cols])

    # Convert scaled arrays back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled_numeric, columns=numeric_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_numeric, columns=numeric_cols, index=X_test.index)

    # Concatenate non-numeric columns back (if any)
    if len(non_numeric_cols) > 0:
        X_train_scaled = pd.concat([X_train_scaled, X_train[non_numeric_cols]], axis=1)
        X_test_scaled = pd.concat([X_test_scaled, X_test[non_numeric_cols]], axis=1)
    # Reorder columns to match original
    X_train_scaled = X_train_scaled[X_train.columns]
    X_test_scaled = X_test_scaled[X_test.columns]
    logger.info("Data normalization complete.")

    # Ensure the output directory exists. This should already be handled by the
    # previous script, but it's good practice for robustness.
    os.makedirs(output_dir, exist_ok=True)

    # Save the two new scaled datasets to the data/processed directory
    logger.info("Saving normalized datasets to data/processed...")
    X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
    logger.info("Normalized datasets saved successfully.")

if __name__ == "__main__":
    # Determine the base directory of the project (e.g., 'examen_dvc')
    # This assumes the script is located in src/data/
    base_dir = Path(__file__).resolve().parents[2]

    # Execute the data normalization function
    normalize_and_save_data(base_dir)
