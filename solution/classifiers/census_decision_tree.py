"""
Decision Tree Classifiers for two U.S Census Income Data
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
import timeit

# a way to get around relative imports outside of this package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from solution.classifiers import helpers
from solution.preprocessors.data_loader import CensusDataLoader

RUN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(RUN_PATH, "data")
CSV_FILENAME = "raw_census_data.csv"
HERE = os.path.abspath(os.path.dirname(__file__))

raw_data_columns = [
    'age', 'workclass', 'fnwgt',
    'education', 'education-num',
    'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'capital-gain',
    'capital-loss', 'hours-per-week',
    'native-country', 'income'
]

derived_feature_columns = [
    'age_num', 'workclass_num', 'marital-status_num',
    'occupation_num', 'relationship_num', 'race_num',
    'sex_num', 'native-country_num', 'income_num'
]

df = pd.read_csv(os.path.join(DATA_PATH, CSV_FILENAME))
df = CensusDataLoader(df).apply_pipeline()

# TODO: add non-pruned results
# These are subject to change based on pre-processing
feature_cols = ['age_num', 'education-num', 'marital-status_Single',
                'hours-per-week', 'capital-gain',
                'capital-loss', 'sex_Male', 'from_united_states']

kfold = KFold(n_splits=5)
tree_cls = DecisionTreeClassifier()

# Plot the learning curve vs train size.
# Helps determine the train vs test split split ratio
helpers.plot_learning_curve_vs_train_size(
    tree_cls,
    df,
    feature_cols,
    'income_num',
    output_location='census_output/dt_num_samples_learning_curve.png'
)

x_train, x_test, y_train, y_test = train_test_split(
    df[feature_cols],
    df['income_num'],
    random_state=0,
    test_size=0.35
)

# Plot the learning curve for max depth vs mean test score
helpers.plot_learning_curve_vs_param(
    tree_cls,
    x_train,
    y_train,
    param_grid={
        'random_state': [0],
        'criterion': ['entropy'],
        'max_depth': range(3, 40),
    },
    cv=5,
    measure_type='mean_test_score',
    output_location='census_output/dt_depth_learning_curve.png',
    param_range=list(range(3, 40))
)

helpers.plot_learning_curve_vs_param(
    tree_cls,
    x_train,
    y_train,
    param_grid={
        'random_state': [0],
        'criterion': ['entropy'],
        'max_leaf_nodes': range(5, 100),
    },
    cv=5,
    measure_type='mean_test_score',
    output_location='census_output/dt_leaf_nodes_learning_curve.png',
    param_range=list(range(5, 100)),
    param_name='Max Leaf Nodes',
)

# Find the best model via GridSearchCV
grid_search = GridSearchCV(
    estimator=tree_cls,
    param_grid={
        'random_state': [0],
        'criterion': ['entropy'],
        'max_depth': range(3, 16),
        'max_leaf_nodes': range(5, 22),
    },
    cv=kfold
)

with open(os.path.join(HERE, 'census_output/dt_no_cv_accuracy.txt'), 'w') as no_cv_accuracy:
    for i in range(3, 20):
        clf = DecisionTreeClassifier(random_state=0, max_depth=i, criterion='entropy')

        clf.fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)
        y_predict = clf.predict(x_test)

        no_cv_accuracy.write('Depth: %s, ' % i)
        no_cv_accuracy.write('Train Accuracy: %s, ' % accuracy_score(y_train, y_train_pred))
        no_cv_accuracy.write('Test Accuracy: %s \n' % accuracy_score(y_test, y_predict))

grid_search.fit(x_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)

# train the best model
best_model = grid_search.best_estimator_
# Time fitting best model
start = timeit.default_timer()
best_model.fit(x_train, y_train)
end = timeit.default_timer()
print('Time to fit:', end-start)
helpers.log_fit_time('CENSUS_DT', end-start)

# Export decision tree to graphviz png
helpers.export_decision_tree_to_file(
    best_model,
    feature_names=feature_cols,
    class_names=['at most 50K', 'more than 50K'],
    output_location=r'census_output/census_decision_tree',
    format='png'
)

# Plot the learning curve vs train size after finding the best model
helpers.plot_learning_curve_vs_train_size(
    best_model,
    df,
    feature_cols,
    'income_num',
    output_location='census_output/dt_best_model_num_samples_learning_curve.png',
    lin_space=np.linspace(0.1, 1.0, num=50)
)

# Predict income with the trained best model
y_pred = best_model.predict(x_test)

helpers.produce_model_performance_summary(
    best_model,
    x_test,
    y_test,
    y_pred,
    grid_search=grid_search,
    output_location='census_output/decision_tree_summary.txt',
    cv=kfold,
    scoring='accuracy'
)
