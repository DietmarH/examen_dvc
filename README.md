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

## Data Splitting Script (`src/data/data_split.py`)
This script is responsible for splitting the raw dataset into training and testing sets. It reads the raw data from `data/raw/raw.csv`, separates the features and the target variable (`silica_concentrate`), and splits the data into training (75%) and testing (25%) sets using a fixed random seed for reproducibility. The resulting datasets (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`) are saved in the `data/processed/` directory. Logging is included to track the process and any errors encountered during execution.

## Data Normalization Script (`src/data/normalize.py`)
This script loads the training and testing feature datasets (`X_train.csv` and `X_test.csv`) from the `data/processed/` directory, applies standard normalization (zero mean, unit variance) to all numeric columns using `StandardScaler` (fitted only on the training set), and saves the normalized datasets as `X_train_scaled.csv` and `X_test_scaled.csv` in the same directory. Non-numeric columns (such as dates or categorical data) are preserved without modification, and the original column order is maintained in the output files. The script logs its progress and errors for easier debugging and reproducibility.

## Hyperparameter Optimization Script (`src/models/grid_search.py`)
This script performs hyperparameter optimization for a `RandomForestRegressor` using `GridSearchCV`. It loads the preprocessed and scaled training data (`X_train_scaled.csv` and `y_train.csv`) from the `data/processed/` directory, removes any non-numeric columns from the feature set, and runs a grid search to find the optimal hyperparameters. The best parameters found are saved as a pickle file (`best_params.pkl`) in the `models/` directory. The script uses 3-fold cross-validation and R² scoring, and logs progress and errors for traceability.

## Model Training Script (`src/models/training.py`)
This script trains a `RandomForestRegressor` model using preprocessed and scaled training data. It loads the feature set (`X_train_scaled.csv`) and target values (`y_train.csv`) from the `data/processed/` directory, loads the best hyperparameters from `models/best_params.pkl`, drops any non-numeric columns from the features, and fits the model. The trained model is then saved as `trained_model.joblib` in the `models/` directory. The script includes robust logging and error handling for traceability and reproducibility.

## Model Evaluation Script (`src/models/evaluate.py`)
This script evaluates a trained `RandomForestRegressor` model using the scaled test dataset. It loads the test features (`X_test_scaled.csv`) and target values (`y_test.csv`) from the `data/processed/` directory, loads the trained model from `models/trained_model.joblib`, and drops any non-numeric columns from the test features to ensure compatibility. The script then generates predictions, calculates evaluation metrics (Mean Squared Error and R² Score), and saves these metrics as `metrics/scores.json`. It also saves the actual and predicted values as `data/predictions.csv`. The script includes robust logging and error handling for traceability and debugging.

# Project Setup Guide

## Setting Up Python, Cloning the Repository, and Creating a Virtual Environment

Follow these steps to prepare your environment for the project:

1. **Update your system and install prerequisites**
   - Open a terminal and run:
     ```bash
     sudo apt update
     sudo apt install -y software-properties-common
     ```
   - This ensures your system is up to date and ready to add new repositories.

2. **Add the Deadsnakes PPA for Python 3.10**
   - This step allows you to install Python 3.10 on Ubuntu 20.04:
     ```bash
     sudo add-apt-repository ppa:deadsnakes/ppa -y
     sudo apt update
     ```

3. **Install Python 3.10 and venv**
   - Install Python 3.10 and the required modules for creating virtual environments:
     ```bash
     sudo apt install -y python3.10 python3.10-venv python3.10-dev
     ```

4. **Verify Python 3.10 Installation**
   - Check that Python 3.10 is installed:
     ```bash
     python3.10 --version
     ```

5. **Clone the GitHub Repository**
   - Replace `<your-username>` with your GitHub username if you forked the repo:
     ```bash
     git clone --single-branch --branch dh-work https://github.com/<your-username>/examen_dvc.git
     cd examen_dvc
     ```

6. **Create and Activate a Virtual Environment**
   - This keeps your project dependencies isolated:
     ```bash
     python3.10 -m venv .venv
     source .venv/bin/activate
     ```
   - You should see your prompt change, indicating the virtual environment is active.

7. **Upgrade pip and Install Required Packages**
   - Upgrade pip and install all dependencies:
     ```bash
     pip install --upgrade pip
     pip install -r requirements.txt
     ```

8. **Verify Python and pip Version in the Virtual Environment**
   - Ensure you are using the correct Python and pip:
     ```bash
     python --version
     pip --version
     ```

---

## 2. Using DVC for Data and Model Versioning

DVC (Data Version Control) helps you manage large files, data sets, machine learning models, and code all in one place.

1. **Initialize DVC in the Project**
   - Run:
     ```bash
     dvc init
     ```
   - This sets up DVC in your project. Commit the changes to git:
     ```bash
     git commit -m "Initialize DVC"
     ```

2. **Set Up DVC Remote Storage**
   - For this example, we use a local directory as remote storage:
     ```bash
     mkdir ../dvc_remote
     dvc remote add -d remote_storage ../dvc_remote
     ```
   - You can use cloud storage (S3, GDrive, etc.) for collaboration.

3. **Remove Data Directory from Git Tracking**
   - Data files should not be tracked by git, only by DVC:
     ```bash
     git rm -r --cached data
     git commit -m "stop tracking data"
     ```

4. **Add Data to DVC Tracking**
   - Track the entire data directory:
     ```bash
     dvc add data
     ```

5. **Commit DVC Changes**
   - Save the DVC tracking files to git:
     ```bash
     git add data.dvc .gitignore .dvc/config
     git commit -m "Track data with DVC"
     ```

6. **Push Data to DVC Remote**
   - Upload your data to the remote storage:
     ```bash
     dvc push
     ```

7. **Import Raw Data and Run the Pipeline**
   - Download the raw data and process it:
     ```bash
     python ./data/import_raw_data.py
     # Run your data processing scripts as needed
     ```

8. **Repeat DVC Add/Commit/Push for New Data**
   - Whenever you add or update data, repeat:
     ```bash
     dvc add data
     dvc commit
     dvc push
     git add data.dvc .gitignore
     git commit -m "Update data"
     git push origin HEAD:dh-work
     ```

---

**Tip:**
- Always activate your virtual environment before running scripts or DVC commands.
- Use `dvc status` to check if your workspace is in sync with the remote.
- To deactivate the virtual environment, run `deactivate`.