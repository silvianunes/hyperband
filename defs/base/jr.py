from common_defs import *


from weka.classifiers import Classifier


space = {
    # 'minNo': hp.uniform('minNo', 1, 5),
    'optimizations': hp.quniform('optimizations', 1, 5, 1),
    'checkerror': hp.choice('ce', (True, False)),
    'pruning': hp.choice('pruning', (True, False)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])

    # L.append("-N")
    # L.append(str(params['minNo']))

    L.append("-O")
    L.append(str(params['optimizations']))

    if params['checkerror'] == False:
        L.append("-E")

    if params['pruning'] == False:
        L.append("-P")

    clf = Classifier(classname="weka.classifiers.rules.JRip", options=L)
    return clf


def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    # L.append("-N")
    # L.append(str(params['minNo']))

    L.append("-O")
    L.append(str(params['optimizations']))

    if params['checkerror'] == False:
        L.append("-E")

    if params['pruning'] == False:
        L.append("-P")

    clf = Classifier(classname="weka.classifiers.rules.JRip", options=L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result
