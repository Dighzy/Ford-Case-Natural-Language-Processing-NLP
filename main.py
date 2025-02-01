import os.path

from src import download_data, utils

path = 'data/raw/full_data_2021_FORD.csv'

if not os.path.isfile(path):
    download_data.download_nhtsa()

utils.check_and_create_env()
