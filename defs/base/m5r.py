from common_defs import *

from weka.classifiers import Classifier


space = {
    'unpruned': hp.choice('n', (True, False)),
    'min_inst': hp.quniform('mi', 1, 64, 1),
    'unsmoothed': hp.choice('u', (True, False)),
    'regression': hp.choice('r', (True, False)),}


def get_params():
    params = sample(space)
    return handle_integers(params)

def get_class(params):
    # pprint(params)

    L = list([])

    if params['unpruned'] == True:
        L.append("-N")

    L.append("-M")
    L.append(str(params['min_inst']))

    if params['unsmoothed'] == True:
        L.append("-U")

    if params['regression'] == False:
        L.append("-R")


    clf = Classifier(classname="weka.classifiers.rules.M5Rules", options=L)
    return clf


def try_params(n_instances, params, train, test, istest):

    n_instances = int(round(n_instances))
    # print "n_instances:", n_instances
    pprint(params)

    L = list([])

    if params['unpruned'] == True:
        L.append("-N")

    L.append("-M")
    L.append(str(params['min_inst']))

    if params['unsmoothed'] == True:
        L.append("-U")

    if params['regression'] == True:
        L.append("-R")


    clf = Classifier(classname="weka.classifiers.rules.M5Rules", options=L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, n_instances)


    return result