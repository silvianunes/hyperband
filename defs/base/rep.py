from common_defs import *

from weka.classifiers import Classifier

space = {
    'minNum': hp.quniform('minNum', 1, 64, 1),
    'minVarProp': hp.uniform('minVarProp', 1e-5, 1e-1),
    'pruning': hp.choice('pruning', (True, False)),
    'maxDepth': hp.quniform('maxDepth', -1, 20, 1),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])
    L.append("-M")
    L.append(str(params['minNum']))

    L.append("-V")
    L.append(str(params['minVarProp']))


    if params['pruning'] == True:
        L.append("-P")

    L.append("-L")
    L.append(str(params['maxDepth']))


    clf = Classifier(classname="weka.classifiers.trees.REPTree", options=L)
    return clf


def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])
    L.append("-M")
    L.append(str(params['minNum']))

    L.append("-V")
    L.append(str(params['minVarProp']))

    if params['pruning'] == True:
        L.append("-P")

    L.append("-L")
    L.append(str(params['maxDepth']))

    clf = Classifier(classname="weka.classifiers.trees.REPTree", options=L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result
