from common_defs import *

from weka.classifiers import Classifier


space = {
    'num_folds': hp.quniform('nf', 2, 5, 1),
    'min_obj': hp.quniform('mo', 1, 64, 1),
    'reduced_prun': hp.choice('r', (True, False)),
    'binary': hp.choice('b', (True, False)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)

def get_class(params):
    # pprint(params)

    L = list([])

    L.append("-M")
    L.append(str(params['min_obj']))

    if params['reduced_prun'] == True:
        L.append("-R")
        L.append("-N")
        L.append(str(params['num_folds']))

    if params['binary'] == True:
        L.append("-B")

    clf = Classifier(classname="weka.classifiers.rules.PART", options=L)
    return clf


def try_params(n_instances, params, train, test, istest):

    n_instances = int(round(n_instances))
    # print "n_instances:", n_instances
    pprint(params)

    L = list([])

    L.append("-M")
    L.append(str(params['min_obj']))

    if params['reduced_prun'] == True:
        L.append("-N")
    else:
        L.append("-N")
        L.append(str(params['num_folds']))

    if params['binary'] == True:
        L.append("-B")

    clf = Classifier(classname="weka.classifiers.rules.PART", options=L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, n_instances)


    return result