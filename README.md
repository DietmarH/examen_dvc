# Examen DVC and Dagshub
In this repository, you will find the proposed architecture for setting up the exam solution.

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
## Raw Data Download Script
The script `import_raw_data.py` is used to download the raw dataset from a specified URL and save it as
`raw.csv` in the directory `./data/raw/`. It includes logging for the download process and ensures the output directory exists. To run the script, simply execute it with Python; it will fetch the default dataset and store it in the appropriate location for further processing.

## Data Splitting Script
The script `src/data/split_data.py` is responsible for splitting the raw dataset into training and testing sets. It reads the raw data from `data/raw/raw.csv`, separates the features and the target variable (`silica_concentrate`), and splits the data into training (75%) and testing (25%) sets using a fixed random seed for reproducibility. The resulting datasets (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`) are saved in the `data/processed/` directory. Logging is included to track the process and any errors encountered during execution.