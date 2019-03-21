from common_defs import *

from weka.classifiers import Classifier


space = {
    'lossFunction': hp.choice('lossFunction', ('0', '2', '4')),
    'learningRate': hp.uniform('learningRate', 0.00001, 0.1),
    'lambda': hp.uniform('lambda', 1e-12, 10),
    'normalize': hp.choice('normalize', (True, False)),
    'missing': hp.choice('missing', (True, False)),

}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    pprint(params)

    L = list([])
    L.append("-F")
    L.append(str(params['lossFunction']))

    L.append("-L")
    L.append(str(params['learningRate']))

    L.append("-R")
    L.append(str(params['lambda']))

    if params['normalize'] == True:
        L.append("-N")

    if params['missing'] == True:
        L.append("-M")

    clf = Classifier(classname="weka.classifiers.functions.SGD", options=L)
    return clf


def try_params(n_instances, params, train, valid, test, istest):

    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])
    L.append("-F")
    L.append(str(params['lossFunction']))

    L.append("-L")
    L.append(str(params['learningRate']))

    L.append("-R")
    L.append(str(params['lambda']))

    if params['normalize'] == True:
        L.append("-N")

    if params['missing'] == True:
        L.append("-M")


    clf = Classifier(classname="weka.classifiers.functions.SGD", options=L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result
