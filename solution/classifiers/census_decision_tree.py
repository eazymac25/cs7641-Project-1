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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

from solution.preprocessors.data_loader import CensusDataLoader

RUN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(RUN_PATH, "data")
CSV_FILENAME = "raw_census_data.csv"


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

feature_cols = ['age_num', 'education-num', 'marital-status_num',
                'occupation_num', 'hours-per-week', 'capital-gain',
                'capital-loss', 'sex_num', 'race_num']

x_train, x_test, y_train, y_test = train_test_split(
    df[feature_cols],
    df['income_num'],
    random_state=0,
    test_size=0.25
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

dots = export_graphviz(
    best_model,
    out_file=None,
    feature_names=feature_cols,
    class_names=['at most 50K', 'more than 50K'],
    filled=True)
graph = graphviz.Source(dots, format='png')
graph.render(r'census_output/census_decision_tree')

# Predict income with the trained best model
y_pred = best_model.predict(x_test)

# Send the output to a file.
with open('census_output/decision_tree_summary.txt', 'w') as output:
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

# Graph the learning curve for the selected model
train_sizes, train_scores, valid_scores = learning_curve(
    DecisionTreeClassifier(**grid_search.best_params_), df[feature_cols], df['income_num'],
    train_sizes=np.linspace(0.1, 1.0),
    cv=5)

plt.figure()
plt.title("Learning Curve")

plt.xlabel("# Training Samples")
plt.ylabel("Accuracy")

plt.plot(train_sizes, np.mean(train_scores, axis=1), color="r", label="Training Set")
plt.plot(train_sizes, np.mean(valid_scores, axis=1), color="g", label="Cross Validation Set")
plt.legend(loc='best')

plt.savefig('census_output/learning_curve.png')

