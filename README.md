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