# meta classifier
from common_defs import *

# loading data
# from load_data import load


from weka.classifiers import Classifier


space = {
    'per_setsize': hp.quniform('ps', 10, 100, 1),
    'numIterations': hp.quniform('ni', 2, 128, 1),
    'seed': hp.quniform('sd', 1, 1, 1),
    'out_of_bag': hp.choice('ur', (True, False)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)
    L = list([])

    L.append("-P")
    L.append(str(params['per_setsize']))

    L.append("-I")
    L.append(str(params['numIterations']))

    L.append("-S")
    L.append(str(params['seed']))

    if params['out_of_bag'] == True and params['per_setsize'] == 100:
        L.append("-O")

    clf = Classifier(classname="weka.classifiers.meta.Bagging", options=L)

    return clf


def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    L.append("-P")
    L.append(str(params['per_setsize']))

    L.append("-I")
    L.append(str(params['numIterations']))

    L.append("-S")
    L.append(str(params['seed']))

    if params['out_of_bag'] == True:
        L.append("-O")

    clf = Classifier(classname="weka.classifiers.meta.Bagging", options=L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result
