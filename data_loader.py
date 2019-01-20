
import os

import requests

DATA_DIRECTORY = "./data"

CENSUS_CSV_FILE_NAME = "raw_census_data.csv"
CENSUS_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
CENSUS_DATA_COLUMNS = [
    'age', 'workclass', 'fnwgt',
    'education', 'education-num',
    'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'capital-gain',
    'capital-loss', 'hours-per-week',
    'native-country', 'flag'
]


def download_census_data_and_save_as_csv():
    if CENSUS_CSV_FILE_NAME in os.listdir(DATA_DIRECTORY):
        raise Exception("File already exists")

    with open(os.path.join(DATA_DIRECTORY, CENSUS_CSV_FILE_NAME), "w") as raw_census_data:
        raw_census_data.write(','.join(CENSUS_DATA_COLUMNS) + '\n')
        raw_census_data.writelines(
            requests.get(CENSUS_DATA_URL).text
        )
