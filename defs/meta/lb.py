# meta classifier
from common_defs import *

# loading data
# from load_data import load


from weka.classifiers import Classifier


space = {
    'resampling': hp.choice('rs', (True, False)),
    'per_setsize': hp.quniform('ps', 10, 100, 1),
    'numIterations': hp.quniform('ni', 2, 64, 1),
    'seed': hp.quniform('sd', 1, 1, 1),
    'threshold': hp.quniform('t', 1, 10, 1),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)
    L = list([])

    # if params['resampling'] == True:
    #     L.append("-Q")

    L.append("-P")
    L.append(str(params['per_setsize']))

    L.append("-I")
    L.append(str(params['numIterations']))

    L.append("-S")
    L.append(str(params['seed']))

    if params['resampling'] == False:
        L.append("-Z")
        L.append(str(params['threshold']))

    clf = Classifier(classname="weka.classifiers.meta.LogitBoost", options=L)

    return clf


def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    # if params['resampling'] == True:
    #     L.append("-Q")

    L.append("-P")
    L.append(str(params['per_setsize']))

    L.append("-I")
    L.append(str(params['numIterations']))

    L.append("-S")
    L.append(str(params['seed']))

    L.append("-Z")
    L.append(str(params['threshold']))

    clf = Classifier(classname="weka.classifiers.meta.LogitBoost", options=L)
    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result
