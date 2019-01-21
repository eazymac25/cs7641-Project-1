import os

import pandas as pd
import requests

RUN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(RUN_PATH, "data")

CENSUS_CSV_FILE_NAME = "raw_census_data.csv"
CENSUS_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
CENSUS_DATA_COLUMNS = [
    'age', 'workclass', 'fnwgt',
    'education', 'education-num',
    'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'capital-gain',
    'capital-loss', 'hours-per-week',
    'native-country', 'income'
]


def download_census_data_and_save_as_csv():
    if CENSUS_CSV_FILE_NAME in os.listdir(DATA_PATH):
        raise Exception("File already exists")

    with open(os.path.join(DATA_PATH, CENSUS_CSV_FILE_NAME), "w") as raw_census_data:
        raw_census_data.write(','.join(CENSUS_DATA_COLUMNS) + '\n')
        raw_census_data.writelines(
            requests.get(CENSUS_DATA_URL).text
        )


class CensusDataLoader(object):

    def __init__(self, df):
        """
        NOTE: self.pipeline shouldn't need to change
        since we are not building an API to run this pipeline.
        However, if the pipeline needs to change,
        a user can update the attribute with more functions
        that take a data frame as input and return a data frame
        as output.
        Parameters:
            df (pandas.DataFrame): data frame that will be operated on
        Returns: void
        """
        self.df = df
        self.pipeline = [
            self.trim_strings,
            self.drop_missing_values,
            self.create_category_num_columns,
            self.bucket_age_column,
        ]

    def apply_pipeline(self):
        """
        Moves through the list of pipeline functions and applies.
        This  assumes idempotent changes. Calling this multiple times
        will result in wasteful ops, but does not change the df.
        Returns:
            self (CensusDataLoader)
        """
        for fxn in self.pipeline:
            self.df = fxn(self.df)
        return self

    @staticmethod
    def trim_strings(df):
        """
        Trim each element if it is a string
        operates against this data frame
        Parameters:
            df (pandas.DataFrame): input data frame. Assumes the data frame
            is of the form of the data frame this CensusDataLoader was initialized.
        Returns:
            df (pandas.DataFrame)
        """
        return df.applymap(
            lambda item: item.strip() if isinstance(item, str) else item)

    @staticmethod
    def drop_missing_values(df):
        """
        Drop missing values which are denoted by '?' in the data set.
        Parameters:
            df (pandas.DataFrame): input data frame. Assumes the data frame
            is of the form of the data frame this CensusDataLoader was initialized.
        Returns:
            df (pandas.DataFrame)
        """
        df = df[df['workclass'] != '?']
        df = df[df['occupation'] != '?']
        df = df[df['native-country'] != '?']
        return df

    @staticmethod
    def create_category_num_columns(df):
        """
        Transform categorical (class) data into a numerical representation.
        Parameters:
            df (pandas.DataFrame): input data frame. Assumes the data frame
            is of the form of the data frame this CensusDataLoader was initialized.
        Returns:
            df (pandas.DataFrame)
        """
        category_maps = {
            'workclass': {key: idx for idx, key in enumerate(set(df['workclass']))},
            'marital-status': {key: idx for idx, key in enumerate(set(df['marital-status']))},
            'occupation': {key: idx for idx, key in enumerate(set(df['occupation']))},
            'relationship': {key: idx for idx, key in enumerate(set(df['relationship']))},
            'race': {key: idx for idx, key in enumerate(set(df['race']))},
            'sex': {key: idx for idx, key in enumerate(set(df['sex']))},
            'native-country': {key: idx for idx, key in enumerate(set(df['native-country']))},
            'income': {'<=50K': 0, '>50K': 1}
        }

        for col, category_map in category_maps.items():
            df[col + '_num'] = df[col].map(category_map)
        return df

    @staticmethod
    def bucket_age_column(df):
        """
        Buckets the age based on the CensusDataLoader._bucket_age_column_helper function.
        Parameters:
            df (pandas.DataFrame): input data frame
        Returns:
            df (pandas.DataFrame): updated data frame
        """
        df['age_num'] = df['age'].apply(
            lambda age: CensusDataLoader._bucket_age_column_helper(age)
        )
        return df

    @staticmethod
    def _bucket_age_column_helper(row_age):
        age_buckets = {
            0: lambda age: age < 20,
            1: lambda age: 20 <= age < 30,
            2: lambda age: 30 <= age < 40,
            3: lambda age: 40 <= age < 50,
            4: lambda age: 50 <= age < 60,
            5: lambda age: age >= 60
        }
        for age_num, evaluator in age_buckets.items():
            if evaluator(row_age):
                return age_num
        raise Exception("No age mapped")


if __name__ == '__main__':
    # print(DATA_PATH)
    # download_census_data_and_save_as_csv()
    dl = CensusDataLoader(pd.read_csv(os.path.join(DATA_PATH, CENSUS_CSV_FILE_NAME)))
    dl.apply_pipeline()

    print(dl.df.head())
