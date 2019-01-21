"""
Decision Tree Classifiers for two U.S Census Income Data
This serves as a simple script and can by run directly.
"""
import os
import sys
# a way to get around relative imports outside of this package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import pandas as pd
import graphviz
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV

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

grid_search = GridSearchCV(
    estimator=tree_cls,
    param_grid={
        'random_state': [0],
        'criterion': ['entropy'],
        'max_depth': range(3, 16),
        'min_samples_leaf': range(100, 1000, 100),
        'min_impurity_decrease': [0.009, 0.1]
    },
    cv=kfold
)

grid_search.fit(x_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)
print(grid_search.cv_results_['mean_test_score'])
print(len(grid_search.cv_results_['mean_test_score']))

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

print(cross_val_score(best_model, x_test, y_test, cv=kfold, scoring='accuracy'))


