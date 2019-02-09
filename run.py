import argparse

from solution.classifiers import (run_census_tree, run_census_nn, run_census_boost,
                                  run_census_svm, run_census_knn)
from solution.classifiers import (run_wine_tree, run_wine_nn, run_wine_boost,
                                  run_wine_svm, run_wine_knn)

parser = argparse.ArgumentParser(description='Run experiments eg python run.py census tree')

parser.add_argument(
    'dataset',
    nargs=1,
    type=str
)

parser.add_argument(
    'model',
    nargs=1,
    type=str
)

args = parser.parse_args()

model_run_map = {
    'census': {
        'tree': run_census_tree,
        'nn': run_census_nn,
        'boost': run_census_boost,
        'svm': run_census_svm,
        'knn': run_census_knn,
    },
    'wine': {
        'tree': run_wine_tree,
        'nn': run_wine_nn,
        'boost': run_wine_boost,
        'svm': run_wine_svm,
        'knn': run_wine_knn,
    }
}

DATASET = args.dataset[0]
MODEL = args.model[0]

models = None
runner = None

try:
    models = model_run_map[DATASET]
except KeyError:
    print("DATASET %s not allowed. Try [census|wine]" % DATASET)


if models:
    try:
        runner = models[MODEL]
    except KeyError:
        print("MODEL %s for DATASET %s does not exist. Try [tree|nn|boost|svm|knn]."
              % (MODEL, DATASET))

if runner:
    print('RUNNING MODEL %s for DATASET %s' % (MODEL, DATASET))
    runner()
