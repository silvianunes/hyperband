from common_defs import *

from weka.classifiers import Classifier


space = {
    'C': hp.uniform('c', 0.5, 1.5),
    'filterType': hp.quniform('filterType', 0, 2, 1),
    'buildCalibrationModels': hp.choice('buildCalibrationModels', (True, False)),
    'kernel': hp.choice('kernel', ('PolyKernel', 'NormalizedPolyKernel', 'Puk', 'RBFKernel')),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])
    L.append("-C")
    L.append(str(params['C']))

    L.append("-N")
    L.append(str(params['filterType']))

    if params['buildCalibrationModels'] == True:
        L.append("-M")

    L.append("-K")
    L.append("weka.classifiers.functions.supportVector." + params['kernel'])


    clf = Classifier(classname="weka.classifiers.functions.SMO", options=L)
    return clf


def try_params(n_instances, params, train, valid, test, istest):

    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])
    L.append("-C")
    L.append(str(params['C']))

    L.append("-N")
    L.append(str(params['filterType']))

    if params['buildCalibrationModels'] == True:
        L.append("-M")

    L.append("-K")
    L.append("weka.classifiers.functions.supportVector." + params['kernel'])


    clf = Classifier(classname="weka.classifiers.functions.SMO", options=L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result
