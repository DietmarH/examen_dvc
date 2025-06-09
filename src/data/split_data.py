import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import logging


def split_and_save_data(base_dir):
    """
    Splits the raw data into training and testing sets and saves them as CSV files.
    Args:
        base_dir (str): The base directory path where the data folders are located.
    Workflow:
        1. Loads the raw data from 'data/raw/raw.csv' within the base directory.
        2. Splits the data into features (X) and target ('silica_concentrate', y).
        3. Splits the dataset into training and testing sets (75% train, 25% test) with a fixed random seed for reproducibility.
        4. Ensures the processed data output directory ('data/processed') exists.
        5. Saves the resulting X_train, X_test, y_train, and y_test as separate CSV files in the processed data directory.
    Raises:
        Logs an error and returns if the raw data file cannot be loaded.
    """
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Path to the raw data file. You might have downloaded this previously into data/raw/
    # If not, you'd need to add a step to download it first or load directly from URL.
    # Based on the problem description, you're working with data after a potential import_raw_data.py [5].
    input_csv_path = os.path.join(base_dir, 'data', 'raw', 'raw.csv') # Assuming 'raw.csv' is the file name

    # Output directory for the processed data [1]
    processed_data_output_dir = os.path.join(base_dir, 'data', 'processed')

    logger.info(f"Loading data from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # The target variable 'silica_concentrate' is in the last column [1, 2]
    X = df.drop(columns=['silica_concentrate'])  # Drop the target column to get features
    y = df['silica_concentrate']   # Use the 'silica_concentrate' column as the target variable

    test_size = 0.25
    random_state = 42
    logger.info(f"Splitting data into training and testing sets with test_size={test_size} and random_state={random_state}")
    # Split the data into training and testing sets
    # A common test_size is 0.2 or 0.25, let's use 0.25 (25% test, 75% train)
    # random_state ensures reproducibility of the split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Ensure the output directory exists [1]
    os.makedirs(processed_data_output_dir, exist_ok=True)

    # Save the four datasets to data/processed [1]
    logger.info("Saving processed datasets...")
    X_train.to_csv(os.path.join(processed_data_output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_data_output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_output_dir, 'y_test.csv'), index=False)

    logger.info(f"Data splitting and saving complete. Datasets saved to {processed_data_output_dir}.")


if __name__ == "__main__":
    # Define the base directory where the script is located
    # This is typically the root of your project, where the 'data' directory is located.
    # Adjust the path as necessary based on your project structure.
    # Assuming the script is located in src/data/split_data.py, we go two levels up to reach the project root.
    base_dir = Path(__file__).resolve().parents[2]
    # Execute the data splitting function
    split_and_save_data(base_dir)
