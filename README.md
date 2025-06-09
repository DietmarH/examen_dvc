# Examen DVC and Dagshub
In this repository you will find the proposed architecture for setting up the exam solution.

```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── processed      
│   │   └── raw       
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src       
│   └── README.md.py       
```
First, you need to fork the repository and then clone it to work on it. The final submission for this exam will be the link to your repository on DagsHub. Make sure to add https://dagshub.com/licence.pedago as a collaborator with read-only access so it can be reviewed.

You can download the dataset from the following link: https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.


# Scripts
## Raw Data Download Script (`import_raw_data.py`)
This script is used to download the raw dataset from a specified URL and save it as
`raw.csv` in the directory `./data/raw/`. It includes logging for the download process and ensures the output directory exists. To run the script, simply execute it with Python; it will fetch the default dataset and store it in the appropriate location for further processing.

## Data Splitting Script (`src/data/split_data.py`)
This script is responsible for splitting the raw dataset into training and testing sets. It reads the raw data from `data/raw/raw.csv`, separates the features and the target variable (`silica_concentrate`), and splits the data into training (75%) and testing (25%) sets using a fixed random seed for reproducibility. The resulting datasets (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`) are saved in the `data/processed/` directory. Logging is included to track the process and any errors encountered during execution.

## Data Normalization Script (`src/data/normalize_data.py`)
This script loads the training and testing feature datasets (`X_train.csv` and `X_test.csv`) from the `data/processed/` directory, applies standard normalization (zero mean, unit variance) to all numeric columns using `StandardScaler` (fitted only on the training set), and saves the normalized datasets as `X_train_scaled.csv` and `X_test_scaled.csv` in the same directory. Non-numeric columns (such as dates or categorical data) are preserved without modification, and the original column order is maintained in the output files. The script logs its progress and errors for easier debugging and reproducibility.

## Hyperparameter Optimization Script (`src/models/find_best_params.py`)
This script performs hyperparameter optimization for a `RandomForestRegressor` using `GridSearchCV`. It loads the preprocessed and scaled training data (`X_train_scaled.csv` and `y_train.csv`) from the `data/processed/` directory, removes any non-numeric columns from the feature set, and runs a grid search to find the optimal hyperparameters. The best parameters found are saved as a pickle file (`best_params.pkl`) in the `models/` directory. The script uses 3-fold cross-validation and R² scoring, and logs progress and errors for traceability.

## Model Training Script (`src/models/train_model.py`)
This script trains a `RandomForestRegressor` model using preprocessed and scaled training data. It loads the feature set (`X_train_scaled.csv`) and target values (`y_train.csv`) from the `data/processed/` directory, loads the best hyperparameters from `models/best_params.pkl`, drops any non-numeric columns from the features, and fits the model. The trained model is then saved as `trained_model.joblib` in the `models/` directory. The script includes robust logging and error handling for traceability and reproducibility.