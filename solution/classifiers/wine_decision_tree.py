"""
Decision Tree Classifiers for  Wine Quality dataData
This serves as a script and can by run directly.
"""
import os
import sys
import timeit

# a way to get around relative imports outside of this package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

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

kfold = KFold(n_splits=5)
tree_cls = DecisionTreeClassifier(random_state=0, criterion='entropy')

# Plot the learning curve vs train size.
# Helps determine the train vs test split split ratio
helpers.plot_learning_curve_vs_train_size(
    tree_cls,
    df,
    feature_cols,
    'quality_num',
    output_location='wine_output/initial_num_samples_learning_curve.png'
)

x_train, x_test, y_train, y_test = train_test_split(
    df[feature_cols],
    df['quality_num'],
    random_state=0,
    test_size=0.2
)

# Plot the learning curve for max depth vs mean test score
helpers.plot_learning_curve_vs_param_train_and_test(
    tree_cls,
    x_train,
    y_train,
    x_test=x_test,
    y_test=y_test,
    param='max_depth',
    param_values=list(range(3, 33)),
    param_name='Max Depth',
    output_location='wine_output/depth_learning_curve.png'
)

# Find the best model via GridSearchCV
grid_search = GridSearchCV(
    estimator=tree_cls,
    param_grid={
        'max_depth': range(3, 16),
        'max_leaf_nodes': range(5, 33),
    },
    cv=kfold
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
helpers.log_fit_time('WINE_DT', end-start)

# Export decision tree to graphviz png
helpers.export_decision_tree_to_file(
    best_model,
    feature_names=feature_cols,
    class_names=['Low Quality', 'High Quality'],
    output_location=r'wine_output/decision_tree',
    format='png'
)

# Plot the learning curve vs train size after finding the best model
helpers.plot_learning_curve_vs_train_size(
    best_model,
    df,
    feature_cols,
    'quality_num',
    output_location='wine_output/best_model_num_samples_learning_curve.png'
)

# Predict income with the trained best model
y_pred = best_model.predict(x_test)

helpers.produce_model_performance_summary(
    best_model,
    x_test,
    y_test,
    y_pred,
    grid_search=grid_search,
    output_location='wine_output/decision_tree_summary.txt',
    cv=kfold,
    scoring='accuracy'
)
