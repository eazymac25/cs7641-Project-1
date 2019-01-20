"""
Decision Tree Classifiers for two data sets:
    - U.S. Census
    - Something else
"""
import os

import numpy as np
import pandas as pd
import requests
import graphviz

from sklearn.tree import DecisionTreeClassifier


class CensusDecisionTree(object):
    raw_data_columns = [
        'age', 'workclass', 'fnwgt',
        'education', 'education-num',
        'marital-status', 'occupation', 'relationship',
        'race', 'sex', 'capital-gain',
        'capital-loss', 'hours-per-week',
        'native-country', 'flag'
    ]

    raw_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    csv_filename = "raw_census_data.csv"

    def __init__(self, raw_df=None):
        self.raw_df = raw_df

    def download_archive_and_save_as_csv(self):
        # TODO: Move this to a top level package and download

        if self.csv_filename in os.listdir("./"):
            raise Exception("File already exists")

        with open(self.csv_filename, "w") as raw_census_data:
            raw_census_data.write(','.join(self.raw_data_columns) + '\n')
            raw_census_data.writelines(
                requests.get(self.raw_data_url).text
            )

    def setup(self):
        """
        Fetches the data from archives and loads to a data frame
        :return: void
        """

        self.raw_df = pd.read_csv(self.csv_filename)


if __name__ == '__main__':

    dt = CensusDecisionTree()
    # dt.download_archive_and_save_as_csv()


