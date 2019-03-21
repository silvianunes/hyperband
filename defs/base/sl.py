from common_defs import *

from weka.classifiers import Classifier


space = {
    'stop': hp.choice('s', (True, False)),
    'weight': hp.choice('w', (0, 1)),
    'aic': hp.choice('a', (True, False)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])

    if params['stop'] == True:
        L.append("-S")

    L.append("-W")
    L.append(str(params['weight']))

    if params['aic'] == True:
        L.append("-A")


    clf = Classifier(classname="weka.classifiers.functions.SimpleLogistic", options=L)

    return clf


def try_params(n_instances, params, train, test, istest):

    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])
    if params['stop'] == True:
        L.append("-S")

    L.append("-W")
    L.append(str(params['weight']))

    if params['aic'] == True:
        L.append("-A")


    clf = Classifier(classname="weka.classifiers.functions.SimpleLogistic", options=L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, n_instances)


    return result
