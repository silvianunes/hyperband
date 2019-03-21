# meta classifier
from common_defs import *

from weka.classifiers import Classifier


space = {
    'debug': hp.choice('d', (True, False)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)
    L = list([])

    if params['debug'] == True:
        L.append("-D")

    clf = Classifier(classname="weka.classifiers.trees.DecisionStump", options=L)

    return clf


def try_params(n_instances, params, train, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    if params['debug'] == True:
        L.append("-D")

    clf = Classifier(classname="weka.classifiers.trees.DecisionStump", options=L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, n_instances)


    return result
