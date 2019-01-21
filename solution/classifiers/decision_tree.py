"""
Decision Tree Classifiers for two data sets:
    - U.S. Census
    - Something else
"""
import os
import sys
# a way to get around relative imports outside of this package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
import pandas as pd
import graphviz
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV

from solution.preprocessors.data_loader import CensusDataLoader

RUN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(RUN_PATH, "data")


class CensusDecisionTree(object):
    raw_data_columns = [
        'age', 'workclass', 'fnwgt',
        'education', 'education-num',
        'marital-status', 'occupation', 'relationship',
        'race', 'sex', 'capital-gain',
        'capital-loss', 'hours-per-week',
        'native-country', 'income'
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
    adf = dt.df

    preprocessor = CensusDataLoader(adf)
    preprocessor.apply_pipeline()
    adf = preprocessor.df

    feature_cols = ['age', 'workclass_num', 'education-num'] #'marital-status_num', 'occupation_num']

    x_train, x_test, y_train, y_test = train_test_split(
        adf[feature_cols],
        adf['income_num'],
        random_state=0,
        test_size=0.25
    )

    tree_cls = DecisionTreeClassifier(random_state=0)

    gd_sr = GridSearchCV(
        estimator=tree_cls,
        param_grid={
            'random_state': [0],
            'criterion': ['gini', 'entropy'],
            'max_depth': range(3, 10)
        },
        cv=5
    )

    gd_sr.fit(x_train, y_train)

    print(gd_sr.best_score_)
    print(gd_sr.best_params_)

    tree_cls = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5, max_leaf_nodes=8)

    tree_cls.fit(x_train, y_train)

    dots = export_graphviz(tree_cls, out_file=None, feature_names=feature_cols, class_names=['less', 'more'], filled=True)
    graph = graphviz.Source(dots, format='png')
    graph.render('census_decision_tree')


