"""
Decision Tree Classifiers for  Wine Quality dataData
This serves as a script and can by run directly.
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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

# We have to do some import magic for this to work on a Mac
# https://github.com/MTG/sms-tools/issues/36
from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

from solution.preprocessors.data_loader import WineDataLoader

RUN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(RUN_PATH, "data")
CSV_FILENAME = "winequality-red.csv"

df = pd.read_csv(os.path.join(DATA_PATH, CSV_FILENAME))
df = WineDataLoader(df).apply_pipeline()

# These are subject to change based on preprocessing
feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

x_train, x_test, y_train, y_test = train_test_split(
    df[feature_cols],
    df['quality_num'],
    random_state=0,
    test_size=0.2
)

kfold = KFold(n_splits=5)
tree_cls = DecisionTreeClassifier()

# Find the best model via GridSearchCV
grid_search = GridSearchCV(
    estimator=tree_cls,
    param_grid={
        'random_state': [0],
        'criterion': ['entropy'],
        'max_depth': range(3, 16),
        'max_leaf_nodes': range(5, 17),
        # 'min_samples_leaf': range(100, 1000, 100),
        # 'min_impurity_decrease': [0.009, 0.1]
    },
    cv=kfold
)

grid_search.fit(x_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)

# train the best model
best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)

# Export decision tree to graphviz png
try:
    dots = export_graphviz(
        best_model,
        out_file=None,
        feature_names=feature_cols,
        class_names=['Low', 'High'],
        filled=True)
    graph = graphviz.Source(dots, format='png')
    graph.render(r'wine_output/wine_decision_tree')
except Exception as e:
    print("Exception using graphviz with error: %s" % e)
    print("Did you install graphviz (sudo apt-get install graphviz)?")
    pass

# Predict income with the trained best model
y_pred = best_model.predict(x_test)

# Send the output of cross validation to a file.
with open('wine_output/decision_tree_summary.txt', 'w') as output:
    output.write('################ GRAPH SEARCH SUMMARY ################\n')

    output.write('BEST SCORE: ' + str(grid_search.best_score_))
    output.write('\n')
    output.write('BEST PARAMS: ' + str(grid_search.best_params_))

    output.write('\n############### PREDICTION SUMMARY ####################\n')
    output.write('CROSS VALIDATION:\n')
    output.write(str(cross_val_score(best_model, x_test, y_test, cv=kfold, scoring='accuracy')))
    output.write('\n')
    output.write('CONFUSION MATRIX:\n')
    output.write(str(confusion_matrix(y_test, y_pred)))
    output.write('\n')
    output.write('CLASSIFICATION REPORT:\n')
    output.write(str(classification_report(y_test, y_pred)))
    output.write('\n')

# Graph the learning curve for number of samples vs accuracy for the best model
train_sizes, train_scores, valid_scores = learning_curve(
    DecisionTreeClassifier(**grid_search.best_params_), df[feature_cols], df['quality_num'],
    train_sizes=np.linspace(0.1, 1.0),
    cv=5)

plt.figure()
plt.title("Learning Curve - Training Set Size")

plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")

plt.plot(train_sizes, np.mean(train_scores, axis=1), color="r", label="Training Set")
plt.plot(train_sizes, np.mean(valid_scores, axis=1), color="g", label="Cross Validation Set")
plt.legend(loc='best')

plt.savefig('wine_output/num_samples_learning_curve.png')

# Plot the learning curve for max depth vs mean test score
grid_search = GridSearchCV(
    estimator=tree_cls,
    param_grid={
        'random_state': [0],
        'criterion': ['entropy'],
        'max_depth': range(3, 16),
    },
    cv=5
)

grid_search.fit(x_train, y_train)

plt.figure()
plt.title("Learning Curve - Depth")
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.plot(list(range(3, 16)), grid_search.cv_results_['mean_test_score'])
plt.savefig("wine_output/depth_learning_curve.png")
