#!/bin/bash

# Install my environment on Ubuntu 20.04

# Ubuntu 20.04 comes with Python 3.8 by default.
# I want to use Python 3.10

# 1. Update System and Install Prerequisites
sudo apt update
sudo apt install -y software-properties-common

#2. Add Deadsnakes PPA for Python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# 3. Install Python 3.10 and python3.10-venv
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# 4. Verify Python 3.10 Installation
python3.10 --version

# 5. Clone the repository
git clone --single-branch --branch dh-work https://github.com/DietmarH/examen_dvc.git

# 6. Create and activate a virtual environment using Python 3.10
cd examen_dvc
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 7. Verify Python Version in the Virtual Environment
#    Both commands should now show Python 3.10 and its corresponding pip version
python --version
pip --version

# 8. Install the required packages  
pip install -r requirements.txt

# 9. Init dvc
dvc init

# 10. Store this information locally on your machine or remotely (e.g., on Google Drive, AWS S3, Azure, Google Cloud)
#     Wew use locally storage for this example
mkdir ../dvc_remote

# 11. Remove all files in the data directory from Git tracking
git rm -r --cached data
git commit -m "stop tracking data"

# 12. Configure DVC remote storage
dvc remote add -d remote_storage ../dvc_remote

# 13. Add the whole data directory to DVC
dvc add data

# 14. Commit the changes to DVC
dvc commit

# 15. Push changes to the remote dvc repository
dvc push

# 16. Push changes to the git repository
git add data.dvc .gitignore
git commit -m "Initial commit with DVC setup"
git push origin HEAD:dh-work

# 17. Repeat the steps 14 to 16 whenever you add new data files to the data directory

# --- To deactivate the virtual environment, run:
# deactivate
