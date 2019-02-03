"""
Neural Net Classifier for two U.S Census Income Data
This serves as a simple script and can by run directly.

We do the following steps:
"""
import os
import sys
import timeit

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
# feature_cols = ['age_num', 'education-num', 'marital-status_Single',
#                 'hours-per-week', 'capital-gain',
#                 'capital-loss']
feature_cols = ['age_num', 'education-num', 'marital-status_Single',
                'hours-per-week', 'capital-gain',
                'capital-loss', 'sex_Male', 'from_united_states']

kfold = KFold(n_splits=5)
cls = MLPClassifier(
    solver='sgd',
    alpha=.001,
    hidden_layer_sizes=(100,),
    random_state=0,
    activation='logistic',
    max_iter=2000)

# Plot the learning curve vs train size.
# Helps determine the train vs test split split ratio
helpers.plot_learning_curve_vs_train_size(
    cls,
    df,
    feature_cols,
    'income_num',
    output_location='census_output/neural_net_num_samples_learning_curve.png'
)

x_train, x_test, y_train, y_test = train_test_split(
    df[feature_cols],
    df['income_num'],
    random_state=0,
    test_size=0.35
)

# Plot the learning curve for max iter vs mean test score
helpers.plot_learning_curve_vs_param_train_and_test(
    MLPClassifier(
        solver='sgd',
        alpha=1e-3,
        hidden_layer_sizes=(100,),
        random_state=0,
        activation='logistic'),
    x_train,
    y_train,
    x_test=x_test,
    y_test=y_test,
    param_grid={
        'max_iter': range(100, 1100, 100),
    },
    cv=5,
    param_name='Max Iterations',
    param_range=list(range(100, 1100, 100)),
    measure_type='mean_test_score',
    output_location='census_output/nn_max_iter_learning_curve.png'
)

# Find the best model via GridSearchCV
grid_search = GridSearchCV(
    estimator=MLPClassifier(solver='sgd', random_state=0, activation='logistic', max_iter=1000),
    param_grid={
        'alpha': [1e-5, 1e-4, 1e-3],
        'hidden_layer_sizes': [(100,), (20, 5)]
    },
    cv=5
)

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
helpers.log_fit_time('CENSUS_NN', end-start)

# Predict income with the trained best model
y_pred = best_model.predict(x_test)

helpers.produce_model_performance_summary(
    best_model,
    x_test,
    y_test,
    y_pred,
    output_location='census_output/neural_net_summary.txt',
    cv=kfold,
    scoring='accuracy'
)
