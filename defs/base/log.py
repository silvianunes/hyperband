from common_defs import *

from weka.classifiers import Classifier


space = {
    'ridge': hp.loguniform('ridge', log(1e-10), log(1)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])

    L.append("-R")
    L.append(str(params['ridge']))

    clf = Classifier(classname="weka.classifiers.functions.Logistic", options=L)
    return clf


def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    L.append("-R")
    L.append(str(params['ridge']))

    clf = Classifier(classname="weka.classifiers.functions.Logistic", options=L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result
