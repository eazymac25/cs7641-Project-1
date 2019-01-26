"""
Decision Tree Classifiers for two U.S Census Income Data
This serves as a simple script and can by run directly.
"""
import os
import sys

# a way to get around relative imports outside of this package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
import pandas as pd
import graphviz
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier

# We have to do some import magic for this to work on a Mac
# https://github.com/MTG/sms-tools/issues/36
from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

from solution.preprocessors.data_loader import CensusDataLoader

RUN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(RUN_PATH, "data")
CSV_FILENAME = "raw_census_data.csv"

df = pd.read_csv(os.path.join(DATA_PATH, CSV_FILENAME))
df = CensusDataLoader(df).apply_pipeline()

# These are subject to change based on preprocessing
feature_cols = ['age_num', 'education-num', 'marital-status_Single',
                'hours-per-week', 'capital-gain',
                'capital-loss', 'sex_Male', 'from_united_states']

x_train, x_test, y_train, y_test = train_test_split(
    df[feature_cols],
    df['income_num'],
    random_state=0,
    test_size=0.35
)
