from common_defs import *

from weka.classifiers import Classifier

space = {
    'binary': hp.choice('b', (True, False)),
    'residuals': hp.choice('r', (True, False)),
    'crossValidated': hp.choice('cv', (True, False)),
    'probabilities': hp.choice('p', (True, False)),
    'min_inst': hp.quniform('mi', 1, 64, 1),
    'weighting': hp.choice('w', (0, hp.uniform('w', 0, 1))),
    'aic': hp.choice('a', (True, False)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])

    if params['binary'] == True:
        L.append("-B")

    if params['residuals'] == True:
        L.append("-R")

    if params['crossValidated'] == True:
        L.append("-C")

    if params['probabilities'] == True:
        L.append("-P")

    L.append("-M")
    L.append(str(params['min_inst']))

    if params['weighting'] != 0 and params['probabilities'] == False:
        L.append("-W")
        L.append(str(params['weighting']))

    L.append("-A")
    L.append(str(params['aic']))

    clf = Classifier(classname="weka.classifiers.trees.LMT", options=L)
    return clf


def try_params(n_instances, params, train, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    if params['binary'] == True:
        L.append("-B")

    if params['residuals'] == True:
        L.append("-R")

    if params['crossValidated'] == True:
        L.append("-C")

    if params['probabilities'] == True:
        L.append("-P")

    L.append("-M")
    L.append(str(params['min_inst']))

    if params['weighting'] != 0:
        L.append("-P")
        L.append(str(params['weighting']))

    L.append("-A")
    L.append(str(params['aic']))

    clf = Classifier(classname="weka.classifiers.trees.LMT", options=L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, n_instances)


    return result
