from common_defs import *

from weka.classifiers import Classifier


space = {
    'useKernelEstimator': hp.choice('useKernelEstimator', (True, False)),
    'useSupervisedDiscretization': hp.choice('useSupervisedDiscretization', (True, False)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])

    if params['useKernelEstimator'] == True:
        L.append("-K")

    if params['useSupervisedDiscretization'] == True and params['useKernelEstimator'] == False:
        L.append("-D")

    clf = Classifier(classname="weka.classifiers.bayes.NaiveBayes", options=L)
    return clf


def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    if params['useKernelEstimator'] == True:
        L.append("-K")

    if params['useSupervisedDiscretization'] == True and params['useKernelEstimator'] == False:
        L.append("-D")

    clf = Classifier(classname="weka.classifiers.bayes.NaiveBayes", options=L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result