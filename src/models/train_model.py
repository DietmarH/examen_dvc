import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib # For saving/loading Python objects like models
import os
import logging
from pathlib import Path


def train_model(base_dir):
    """
    Trains a RandomForestRegressor model using preprocessed and scaled training data, 
    and saves the trained model to disk.

    This function performs the following steps:
        1. Loads scaled training features (`X_train_scaled.csv`) and target values (`y_train.csv`)
           from the processed data directory.
        2. Loads the best hyperparameters for the RandomForestRegressor from a pickle file (`best_params.pkl`)
           in the models directory.
        3. Initializes a RandomForestRegressor with the loaded parameters (ensuring reproducibility by setting
           `random_state=42` if not present in the parameters).
        4. Drops any non-numeric columns from the training features to ensure compatibility with scikit-learn.
        5. Trains the model on the provided data.
        6. Saves the trained model as a `.joblib` file in the models directory.

        base_dir (str): The base directory of the project. The function expects the following structure:
            - Processed training data at: <base_dir>/data/processed/X_train_scaled.csv and y_train.csv
            - Best parameters at: <base_dir>/models/best_params.pkl
            - Trained model will be saved to: <base_dir>/models/trained_model.joblib

    Returns:
        None

    Raises:
        Logs errors and exits gracefully if required files are missing or if unexpected exceptions occur.
    """


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)


    # Define the path to the data/processed directory for input
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')

    # Define the path to the models directory for input/output
    models_dir = os.path.join(base_dir, 'models')

    logger.info(f"Attempting to load scaled training data from: {processed_data_dir}")
    try:
        # Load the scaled training features and target
        X_train_scaled = pd.read_csv(os.path.join(processed_data_dir, 'X_train_scaled.csv'))
        y_train = pd.read_csv(os.path.join(processed_data_dir, 'y_train.csv'))
        # .values.ravel() converts DataFrame to 1D array, as expected by scikit-learn models
        y_train = y_train.values.ravel()
        logger.info("Scaled training data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Input training data files (X_train_scaled.csv or y_train.csv) not found in {processed_data_dir}.\nPlease ensure the 'Data Splitting' and 'Data Normalization' scripts have been run and saved the files correctly.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading training data: {e}")
        return

    best_params_path = os.path.join(models_dir, 'best_params.pkl')
    logger.info(f"Attempting to load best parameters from: {best_params_path}")
    try:
        # Load the best parameters found by GridSearchCV
        best_params = joblib.load(best_params_path)
        logger.info(f"Best parameters loaded successfully:\n{best_params}")
    except FileNotFoundError:
        logger.error(f"Error: 'best_params.pkl' not found in {models_dir}.\nPlease ensure the 'GridSearch for Best Parameters' script has been run and saved the parameters correctly.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading best parameters: {e}")
        return

    logger.info("Initializing RandomForestRegressor with best parameters...")
    # Initialize the RandomForestRegressor with the best parameters
    # Set a fixed random_state for reproducibility if not already in best_params
    if 'random_state' not in best_params:
        model = RandomForestRegressor(random_state=42, **best_params)
    else:
        model = RandomForestRegressor(**best_params)

    # Drop non-numeric columns (e.g., 'date') before training
    non_numeric_cols = X_train_scaled.select_dtypes(exclude=['number']).columns
    X_train_for_fit = X_train_scaled.drop(columns=non_numeric_cols)

    logger.info("Training the model...")
    # Train the model
    model.fit(X_train_for_fit, y_train)
    logger.info("Model training complete.")

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Save the trained model
    trained_model_path = os.path.join(models_dir, 'trained_model.joblib')
    joblib.dump(model, trained_model_path)
    logger.info(f"Trained model saved to: {trained_model_path}")


if __name__ == "__main__":
    # Determine the base directory of the project (e.g., 'examen_dvc')
    # This assumes the script is located in src/models/
    base_dir = Path(__file__).resolve().parents[2]

    # Execute the function to train and save the model
    train_model(base_dir)
