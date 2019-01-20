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

RUN_PATH = os.path.dirname(os.getcwd())
DATA_PATH = os.path.join(RUN_PATH, "data")


class CensusDecisionTree(object):
    raw_data_columns = [
        'age', 'workclass', 'fnwgt',
        'education', 'education-num',
        'marital-status', 'occupation', 'relationship',
        'race', 'sex', 'capital-gain',
        'capital-loss', 'hours-per-week',
        'native-country', 'flag'
    ]

    csv_filename = "raw_census_data.csv"

    def __init__(self, df=None):
        self.df = self._get_initial_df() if df is None else df

    def _get_initial_df(self):
        """
        Loads the data from data directory into a data frame
        :return: pandas.DataFrame
        """
        return pd.read_csv(
            os.path.join(DATA_PATH, self.csv_filename)
        )


if __name__ == '__main__':

    dt = CensusDecisionTree()
    # dt.download_archive_and_save_as_csv()


