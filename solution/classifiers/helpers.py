import os
import sys
import timeit
import inspect

# a way to get around relative imports outside of this package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
import graphviz
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import export_graphviz
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


def timer(func):
    def func_timer(*args, **kwargs):
        start = timeit.default_timer()
        parent_idx = 1
        results = func(*args, **kwargs)
        end = timeit.default_timer()
        with open('times.txt', 'a') as time_results:
            time_results.write('Originating Classifier: ' +
                               str(os.path.basename(
                                   os.path.realpath(inspect.getfile(inspect.stack()[parent_idx][0])))) + '\n')
            time_results.write(
                '\tOriginating method: ' + str(func.__name__) + ' took ' + str(end - start) + ' seconds\n')
        print(func.__name__, 'Elapsed', end - start)
        return results

    return func_timer


def log_fit_time(model_origin, total_time):
    with open('times.txt', 'a') as time_results:
        time_results.write(model_origin.upper() + ' FIT TIME: ' + str(total_time) + '\n')


def produce_model_performance_summary(best_model, x_test, y_test, y_pred,
                                      output_location, grid_search=None, cv=5, scoring='accuracy'):
    with open(output_location, 'w') as output:
        if grid_search:
            output.write('################ GRAPH SEARCH SUMMARY ################\n')

            output.write('BEST SCORE: ' + str(grid_search.best_score_))
            output.write('\n')
            output.write('BEST PARAMS: ' + str(grid_search.best_params_))

        output.write('\n############### PREDICTION SUMMARY ####################\n')
        output.write('CROSS VALIDATION:\n')
        output.write(str(cross_val_score(best_model, x_test, y_test, cv=cv, scoring=scoring)))
        output.write('\n')
        output.write('CONFUSION MATRIX:\n')
        output.write(str(confusion_matrix(y_test, y_pred)))
        output.write('\n')
        output.write('CLASSIFICATION REPORT:\n')
        output.write(str(classification_report(y_test, y_pred)))
        output.write('\n')


@timer
def plot_learning_curve_vs_train_size(classifier, dataframe, feature_cols, target_col, output_location,
                                      training_label='Training Set', validation_label='Cross Validation Set',
                                      lin_space=np.linspace(0.1, 1.0, num=10)):
    train_sizes, train_scores, validation_scores = learning_curve(
        classifier,
        dataframe[feature_cols],
        dataframe[target_col],
        train_sizes=lin_space,
        cv=5
    )

    plt.figure()
    plt.title("Learning Curve - Training Set Size")

    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")

    plt.plot(train_sizes, np.mean(train_scores, axis=1), color="r", label=training_label)
    plt.plot(train_sizes, np.mean(validation_scores, axis=1), color="g", label=validation_label)
    plt.legend(loc='best')

    plt.savefig(output_location)


@timer
def plot_learning_curve_vs_param(classifier, x_train, y_train, param_grid={}, param_name='Depth',
                                 param_range=list(range(3, 16)), cv=5,
                                 measure_type='mean_test_score', output_location=''):
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=cv
    )

    grid_search.fit(x_train, y_train)

    plt.figure()
    plt.title("Learning Curve - %s" % param_name)
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.plot(param_range, grid_search.cv_results_[measure_type])
    plt.savefig(output_location)


@timer
def export_decision_tree_to_file(model, feature_names=[], class_names=[], output_location=r'', format='png'):
    try:
        dots = export_graphviz(
            model,
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True)
        graph = graphviz.Source(dots, format=format)
        graph.render(output_location)
    except Exception as e:
        print("Exception using graphviz with error: %s" % e)
        print("Did you install graphviz (sudo apt-get install graphviz)?")
