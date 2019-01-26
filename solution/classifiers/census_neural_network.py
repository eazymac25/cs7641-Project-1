"""
Neural Net Classifier for two U.S Census Income Data
This serves as a simple script and can by run directly.

We do the following steps:
    1. Pre-process the data set
    2. Plot learning curve for data set train size vs accuracy
    3. Split into training set and test set
    4. Plot learning curve vs max depth
    5. Do grid search against all relevant params and find best model
    6. Fit to best model
    7. Predict from best model
    8. Produce prediction summary
"""
import os
import sys

# a way to get around relative imports outside of this package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from solution.classifiers import helpers
from solution.preprocessors.data_loader import CensusDataLoader

RUN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(RUN_PATH, "data")
CSV_FILENAME = "raw_census_data.csv"

df = pd.read_csv(os.path.join(DATA_PATH, CSV_FILENAME))
df = CensusDataLoader(df).apply_pipeline()

# These are subject to change based on pre-processing
feature_cols = ['age_num', 'education-num', 'marital-status_Single',
                'hours-per-week', 'capital-gain',
                'capital-loss']

kfold = KFold(n_splits=5)
cls = MLPClassifier(solver='sgd', alpha=.001, hidden_layer_sizes=(20, 5), random_state=0, activation='logistic', max_iter=2000)

# Plot the learning curve vs train size.
# Helps determine the train vs test split split ratio
helpers.plot_learning_curve_vs_train_size(
    cls,
    df,
    feature_cols,
    'income_num',
    output_location='census_output/neural_net_num_samples_learning_curve.png'
)

print('b')

x_train, x_test, y_train, y_test = train_test_split(
    df[feature_cols],
    df['income_num'],
    random_state=0,
    test_size=0.35
)

# Plot the learning curve for max depth vs mean test score
# helpers.plot_learning_curve_vs_param(
#     cls,
#     x_train,
#     y_train,
#     param_grid={
#         'random_state': [0],
#         'solver': ['sgd'],
#         'activation': ['logistic'],
#         'hidden_layer_sizes': ''
#     },
#     cv=5,
#     measure_type='mean_test_score',
#     output_location='census_output/neural_net_depth_learning_curve.png'
# )

# Find the best model via GridSearchCV
grid_search = GridSearchCV(
    estimator=cls,
    param_grid={
        'solver': ['sgd'],
        'alpha': [1e-5],
        'hidden_layer_sizes': [(20, 5)],
        'random_state': [0],
        'activation': ['logistic']
    },
    cv=kfold
)

grid_search.fit(x_train, y_train)

# print(grid_search.best_score_)
# print(grid_search.best_params_)

# train the best model
best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)

# Predict income with the trained best model
y_pred = cls.predict(x_test)

helpers.produce_model_performance_summary(
    cls,
    x_test,
    y_test,
    y_pred,
    output_location='census_output/neural_net_summary.txt',
    cv=kfold,
    scoring='accuracy'
)
