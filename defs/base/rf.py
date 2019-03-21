from common_defs import *

from weka.classifiers import Classifier

space = {
    'numInterations': hp.quniform('ni', 2, 256, 1),
    'numattr': hp.quniform('na', 0, 32, 1),
    'depth': hp.quniform('d', 0, 20, 1),

}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])

    L.append("-I")
    L.append(str(params['numInterations']))

    L.append("-K")
    L.append(str(params['numattr']))

    L.append("-depth")
    L.append(str(params['depth']))


    clf = Classifier(classname="weka.classifiers.trees.RandomForest", options=L)
    return clf


def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    # data = load(directory)

    L = list([])

    L.append("-I")
    L.append(str(params['numInterations']))

    L.append("-K")
    L.append(str(params['numattr']))

    L.append("-depth")
    L.append(str(params['depth']))


    clf = Classifier(classname="weka.classifiers.trees.RandomForest", options=L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result