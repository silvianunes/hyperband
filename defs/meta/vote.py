from common_defs import *

from weka.classifiers import Classifier

space = {
    'rule': hp.choice('r', ('AVG', 'PROD', 'MAJ', 'MIN', 'MAX')),
    'seed': hp.quniform('s', 1, 1, 1),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])

    L.append("-R")
    L.append(params['rule'])

    L.append("-S")
    L.append(str(params['seed']))


    clf = Classifier(classname="weka.classifiers.meta.Vote", options=L)
    return clf


def try_params(n_instances, params, train, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    L.append("-R")
    L.append(params['rule'])

    L.append("-S")
    L.append(str(params['seed']))


    clf = Classifier(classname="weka.classifiers.meta.Vote", options=L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, n_instances)


    return result
