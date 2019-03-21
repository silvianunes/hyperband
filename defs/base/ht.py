from common_defs import *

from weka.classifiers import Classifier


space = {
    'leaf': hp.quniform('leaf', 0, 2, 1),
    'splitCriterion': hp.quniform('splitCriterion', 0, 1, 1),
    'splitConfidence': hp.uniform('splitConfidence', 0.0001, 0.01),
    'hoeffdingTieThreshold': hp.uniform('hoeffdingTieThreshold', 0, 0.1),
    'minimumFractionOfWeightInfoGain': hp.uniform('minimumFractionOfWeightInfoGain', 0, 0.1),
    'gracePeriod': hp.uniform('gracePeriod', 0, 100),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)
    L = list([])
    L.append("-L")
    L.append(str(params['leaf']))

    L.append("-S")
    L.append(str(params['splitCriterion']))

    L.append("-E")
    L.append(str(params['splitConfidence']))

    L.append("-H")
    L.append(str(params['hoeffdingTieThreshold']))

    L.append("-M")
    L.append(str(params['minimumFractionOfWeightInfoGain']))

    L.append("-G")
    L.append(str(params['gracePeriod']))


    clf = Classifier(classname="weka.classifiers.trees.HoeffdingTree", options=L)
    return clf

def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])
    L.append("-L")
    L.append(str(params['leaf']))

    L.append("-S")
    L.append(str(params['splitCriterion']))

    L.append("-E")
    L.append(str(params['splitConfidence']))

    L.append("-H")
    L.append(str(params['hoeffdingTieThreshold']))

    L.append("-M")
    L.append(str(params['minimumFractionOfWeightInfoGain']))

    L.append("-G")
    L.append(str(params['gracePeriod']))


    clf = Classifier(classname="weka.classifiers.trees.HoeffdingTree", options=L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result