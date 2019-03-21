# meta classifier
from common_defs import *

# loading data
# from load_data import load


from weka.classifiers import Classifier


space = {
    'numIterations': hp.quniform('ni', 2, 64, 1),
    'seed': hp.quniform('sd', 1, 1, 1),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)
    L = list([])

    L.append("-I")
    L.append(str(params['numIterations']))

    L.append("-S")
    L.append(str(params['seed']))

    clf = Classifier(classname="weka.classifiers.meta.RandomCommittee", options=L)

    return clf


def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])


    L.append("-I")
    L.append(str(params['numIterations']))

    L.append("-S")
    L.append(str(params['seed']))

    clf = Classifier(classname="weka.classifiers.meta.RandomCommittee", options=L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result
