import os
import requests
import logging
from pathlib import Path


def download_raw_data(url, base_dir):
    """
    Download a raw data file from a specified URL and save it to a 'data/raw' directory under the given base directory.

    Args:
        url (str): The URL from which to download the raw data file.
        base_dir (str or Path): The base directory where the 'data/raw' folder will be created and the file will be saved.

    Raises:
        requests.HTTPError: If the HTTP request for downloading the file fails.

    The function logs the download process and ensures the output directory exists before saving the file as 'raw.csv'.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Define the output directory
    output_dir = os.path.join(base_dir, 'data', 'raw_data')
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'raw.csv')

    logger.info(f"Downloading raw data from {url}")

    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        f.write(response.content)
    logger.info(f"File saved to {output_path}")


if __name__ == "__main__":
    # Define URL and output directory
    url = 'https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv'
    base_dir = Path(__file__).resolve().parents[2]

    download_raw_data(url, base_dir)
