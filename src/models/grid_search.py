import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib # For saving/loading Python objects like dictionaries
import os
import logging
from pathlib import Path

def find_best_params(base_dir):
    """
    Performs hyperparameter optimization for a RandomForestRegressor using GridSearchCV.

    This function loads preprocessed and scaled training data from the specified base directory,
    removes any non-numeric columns from the feature set, and performs a grid search to find the
    optimal hyperparameters for a RandomForestRegressor. The best parameters found are then saved
    as a pickle (.pkl) file in the models directory.

        base_dir (str): The root directory containing the 'data/processed' folder with the scaled
                        training data and the 'models' folder where the best parameters will be saved.

    Workflow:
        1. Loads 'X_train_scaled.csv' and 'y_train.csv' from 'data/processed' within base_dir.
        2. Removes non-numeric columns from the feature set.
        3. Defines a parameter grid for RandomForestRegressor, dynamically adjusting 'max_features'
           based on the number of input features.
        4. Runs GridSearchCV with 3-fold cross-validation and R^2 scoring to find the best parameters.
        5. Saves the best parameters as 'best_params.pkl' in the 'models' directory.

    Logging:
        - Logs progress and errors at various stages for traceability.

    Raises:
        - Logs and exits gracefully if input files are missing or if unexpected errors occur during data loading.

    Returns:
        None. The function saves the best parameters to disk and logs the process.
    """


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Define the path to the data/processed directory for input
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')

    # Define the path to the models directory for output
    models_dir = os.path.join(base_dir, 'models')

    logger.info(f"Attempting to load scaled training data from: {processed_data_dir}")
    try:
        # Load the scaled training features and target
        X_train_scaled = pd.read_csv(os.path.join(processed_data_dir, 'X_train_scaled.csv'))
        y_train = pd.read_csv(os.path.join(processed_data_dir, 'y_train.csv'))
        logger.info("Scaled training data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Input files (X_train_scaled.csv or y_train.csv) not found in {processed_data_dir}.\nPlease ensure the 'Data Splitting' and 'Data Normalization' scripts have been run and saved the files correctly.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        return

    logger.info("Defining the regression model and parameter grid for GridSearch...")
    # Initialize the RandomForestRegressor
    # Use a fixed random_state for reproducibility
    model = RandomForestRegressor(random_state=42)

    # Remove non-numeric columns (e.g., 'date') for training
    non_numeric_cols = X_train_scaled.select_dtypes(exclude=['number']).columns
    X_train_for_grid = X_train_scaled.drop(columns=non_numeric_cols)

    # Dynamically set param_grid based on input data shape
    n_features = X_train_for_grid.shape[1]
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2', max(1, n_features // 2), n_features],
        'min_samples_split': [2, 3, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_depth': [None, 5, 10, 20]
    }

    # Initialize GridSearchCV
    # cv=3 means 3-fold cross-validation
    # n_jobs=-1 uses all available CPU cores
    # scoring can be 'neg_mean_squared_error', 'r2', etc. 'r2' is good for regression
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='r2')

    logger.info("Starting GridSearch to find the best parameters...")
    # Fit GridSearchCV to the training data
    grid_search.fit(X_train_for_grid, y_train.values.ravel()) # .values.ravel() to convert y_train DataFrame to 1D array

    # Get the best parameters
    best_params = grid_search.best_params_
    logger.info(f"GridSearch complete. Best parameters found:\n{best_params}")

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Save the best parameters to a .pkl file
    best_params_path = os.path.join(models_dir, 'best_params.pkl')
    joblib.dump(best_params, best_params_path)
    logger.info(f"Best parameters saved to: {best_params_path}")

if __name__ == "__main__":
    # Determine the base directory of the project (e.g., 'examen_dvc')
    # This assumes the script is located in src/models/
    base_dir = Path(__file__).resolve().parents[2]

    # Execute the function to find and save best parameters
    find_best_params(base_dir)

