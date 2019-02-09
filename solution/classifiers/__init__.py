

__all__ = [
    'run_census_tree',
    'run_census_nn',
    'run_census_boost',
    'run_census_svm',
    'run_census_knn',
    'run_wine_tree',
    'run_wine_nn',
    'run_wine_boost',
    'run_wine_svm',
    'run_wine_knn',
]


def run_census_tree():
    from . import census_decision_tree


def run_census_nn():
    from . import census_neural_network


def run_census_boost():
    from . import census_boosting


def run_census_svm():
    from . import census_svm


def run_census_knn():
    from . import census_knn


def run_wine_tree():
    from . import wine_decision_tree


def run_wine_nn():
    from . import wine_neural_network


def run_wine_boost():
    from . import wine_boosting


def run_wine_svm():
    from . import wine_svm


def run_wine_knn():
    from . import wine_knn
