import pandas as pd
import joblib # For loading the trained model
import json   # For saving metrics
import os     # For path manipulation
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score # For regression metrics


def evaluate_and_predict_model(base_dir):
    """
    Loads a trained machine learning model and scaled test data, evaluates the model's performance,
    saves evaluation metrics, and generates predictions on the test set.

    This function performs the following steps:
        1. Loads the scaled test features and target values from the processed data directory.
        2. Loads the trained model from the models directory.
        3. Drops any non-numeric columns from the test features to ensure compatibility with the model.
        4. Uses the model to predict target values for the test set.
        5. Calculates evaluation metrics (Mean Squared Error and R^2 Score) for the predictions.
        6. Saves the evaluation metrics as a JSON file in the metrics directory.
        7. Saves the actual and predicted values as a CSV file in the data directory.

        base_dir (str): The base directory of the project. Subdirectories for processed data,
                        models, metrics, and output data are resolved relative to this path.

    Raises:
        FileNotFoundError: If required data or model files are missing.
        Exception: For any unexpected errors during data loading, model loading, or prediction.

    Outputs:
        - metrics/scores.json: Contains evaluation metrics (MSE and R^2).
        - data/predictions.csv: Contains actual and predicted target values for the test set.

    Logging:
        Logs progress and errors at various stages for traceability and debugging.
    """


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)


    # Define paths based on the project structure
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    metrics_dir = os.path.join(base_dir, 'metrics')
    # Predictions will be saved in the root 'data' directory as a new dataset [1]
    data_output_dir = os.path.join(base_dir, 'data')


    logger.info(f"Attempting to load scaled test data from: {processed_data_dir}")
    try:
        # Load the scaled test features and target
        X_test_scaled = pd.read_csv(os.path.join(processed_data_dir, 'X_test_scaled.csv'))
        y_test = pd.read_csv(os.path.join(processed_data_dir, 'y_test.csv'))
        # .values.ravel() converts DataFrame to 1D array, as expected by scikit-learn models
        y_test = y_test.values.ravel()
        logger.info("Scaled test data loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Error loading test data: {e}. Please ensure 'Data Splitting' and 'Data Normalization' scripts have been run.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading test data: {e}")
        return

    # Load the trained model from gbr_model.pkl instead of trained_model.joblib
    gbr_model_path = os.path.join(models_dir, 'gbr_model.pkl')
    logger.info(f"Attempting to load trained model from: {gbr_model_path}")
    try:
        model = joblib.load(gbr_model_path)
        logger.info("Trained model loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Error loading trained model: {e}. Please ensure the 'Model Training' script has been run.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the model: {e}")
        return

    # Drop non-numeric columns (e.g., 'date') before prediction
    non_numeric_cols = X_test_scaled.select_dtypes(exclude=['number']).columns
    X_test_for_pred = X_test_scaled.drop(columns=non_numeric_cols)

    logger.info("Making predictions on the test set...")
    # Make predictions on the scaled test set
    y_pred = model.predict(X_test_for_pred)
    logger.info("Predictions made successfully.")

    logger.info("Calculating evaluation metrics...")
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'mean_squared_error': mse,
        'r2_score': r2
    }
    logger.info(f"Evaluation metrics calculated:\n{metrics}")

    # Ensure the metrics directory exists
    os.makedirs(metrics_dir, exist_ok=True)
    logger.info(f"Ensured metrics directory exists: {metrics_dir}")

    # Save the evaluation metrics to a JSON file
    scores_path = os.path.join(metrics_dir, 'scores.json')
    with open(scores_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Evaluation metrics saved to: {scores_path}")

    logger.info("Saving predictions...")
    # Create a DataFrame for predictions
    # You might want to include y_test alongside y_pred for comparison, or X_test_scaled for context
    predictions_df = pd.DataFrame({'actual_silica_concentrate': y_test, 'predicted_silica_concentrate': y_pred})

    # Ensure the data directory exists (root 'data' directory, not 'data/processed')
    os.makedirs(data_output_dir, exist_ok=True)

    # Save predictions to a CSV file in the 'data' directory
    predictions_path = os.path.join(data_output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to: {predictions_path}")

if __name__ == "__main__":
    # Determine the base directory of the project (e.g., 'examen_dvc')
    # This assumes the script is located in src/models/
    base_dir = Path(__file__).resolve().parents[2]

    # Execute the function to evaluate the model and save predictions
    evaluate_and_predict_model(base_dir)
