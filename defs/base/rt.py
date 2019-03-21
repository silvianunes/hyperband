from common_defs import *

from weka.classifiers import Classifier

space = {
    'minInst': hp.quniform('mi', 1, 64, 1),
    'numAttr': hp.choice('na', (0, hp.quniform('kk', 2, 32, 1))),
    'depth': hp.choice('de', (0, hp.quniform('dd', 2, 20, 1))),
    'folds': hp.choice('fd', (0, hp.quniform('nn', 2, 4, 1))),
    'unclass': hp.choice('uc', (True, False)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])

    L.append("-M")
    L.append(str(params['minInst']))

    L.append("-K")
    L.append(str(params['numAttr']))

    L.append("-depth")
    L.append(str(params['depth']))

    L.append("-N")
    L.append(str(params['folds']))

    if params['unclass'] == True:
        L.append("-U")


    clf = Classifier(classname="weka.classifiers.trees.RandomTree", options=L)
    return clf


def try_params(n_instances, params, train, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    L.append("-M")
    L.append(str(params['minInst']))

    L.append("-K")
    L.append(str(params['numAttr']))

    L.append("-depth")
    L.append(str(params['depth']))

    L.append("-N")
    L.append(str(params['folds']))

    if params['unclass'] == True:
        L.append("-U")


    clf = Classifier(classname="weka.classifiers.trees.RandomTree", options=L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, n_instances)


    return result
