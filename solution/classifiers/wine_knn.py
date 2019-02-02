import os
import sys
import timeit

# a way to get around relative imports outside of this package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from solution.classifiers import helpers
from solution.preprocessors.data_loader import WineDataLoader

RUN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(RUN_PATH, "data")
CSV_FILENAME = "winequality-red.csv"

df = pd.read_csv(os.path.join(DATA_PATH, CSV_FILENAME))
df = WineDataLoader(df).apply_pipeline()

# These are subject to change based on preprocessing
feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

kfold = KFold(n_splits=3)
cls = cls = KNeighborsClassifier()

# Plot the learning curve vs train size.
# Helps determine the train vs test split split ratio
helpers.plot_learning_curve_vs_train_size(
    cls,
    df,
    feature_cols,
    'quality_num',
    output_location='wine_output/knn_samples_learning_curve.png'
)

x_train, x_test, y_train, y_test = train_test_split(
    df[feature_cols],
    df['quality_num'],
    random_state=0,
    test_size=0.35
)

# Plot the learning curve for max iter vs mean test score
helpers.plot_learning_curve_vs_param(
    KNeighborsClassifier(),
    x_train,
    y_train,
    param_grid={
        'n_neighbors': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    cv=3,
    param_name='N Neighbors',
    param_range=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    measure_type='mean_test_score',
    output_location='wine_output/knn_n_neighbors_learning_curve.png'
)

# Find the best model via GridSearchCV
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid={
        'n_neighbors': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    cv=3
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
print('Time to fit:', end - start)
helpers.log_fit_time('WINE_KNN', end - start)

# Predict quality with the trained best model
y_pred = best_model.predict(x_test)

helpers.produce_model_performance_summary(
    best_model,
    x_test,
    y_test,
    y_pred,
    output_location='wine_output/knn_summary.txt',
    cv=3,
    scoring='accuracy'
)
