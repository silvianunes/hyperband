from common_defs import *

from weka.classifiers import Classifier


space = {
    'learningRate': hp.uniform('learningRate', 0.1, 1),
    'momentum': hp.uniform('momentum', 0.1, 1),
    'nominalToBinaryFilter': hp.choice('nominalToBinaryFilter', (True, False)),
    'hiddenLayers': hp.choice('hl', ('a', 'i', 'o', 't')),
    'normalizeNumClasses': hp.choice('normalizeNumClasses', (True, False)),
    'reset': hp.choice('reset', (True, False)),
    'decay': hp.choice('decay', (True, False)),
    'seed': hp.quniform('learningRate', 1, 1, 1),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])
    L.append("-L")
    L.append(str(params['learningRate']))

    L.append("-M")
    L.append(str(params['momentum']))


    if params['nominalToBinaryFilter'] == True:
        L.append("-B")

    L.append("-H")
    L.append(str(params['hiddenLayers']))

    if params['normalizeNumClasses'] == True:
        L.append("-C")

    if params['reset'] == True:
        L.append("-R")

    if params['decay'] == True:
        L.append("-D")

    L.append("-S")
    L.append(str(params['seed']))

    clf = Classifier(classname="weka.classifiers.functions.MultilayerPerceptron", options=L)
    return clf


def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])
    L.append("-L")
    L.append(str(params['learningRate']))

    L.append("-M")
    L.append(str(params['momentum']))


    if params['nominalToBinaryFilter'] == True:
        L.append("-B")

    L.append("-H")
    L.append(str(params['hiddenLayers']))

    if params['normalizeNumClasses'] == True:
        L.append("-C")

    if params['reset'] == True:
        L.append("-R")

    if params['decay'] == True:
        L.append("-D")

    L.append("-S")
    L.append(str(params['seed']))

    clf = Classifier(classname="weka.classifiers.functions.MultilayerPerceptron", options=L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result