from common_defs import *

from weka.classifiers import Classifier


space = {
    'K': hp.quniform('k', 1, 64, 1),
    'distanceWeighting': hp.choice('distanceWeighting', ('No distance weighting', 'Weight by 1/distance', 'Weight by 1-distance')),
    'meanSquared': hp.choice('meanSquared', (True, False)),
    'crossValidated': hp.choice('cv', (True, False)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])
    L.append("-K")
    L.append(str(params['K']))

    if params['distanceWeighting'] == 'Weight by 1/distance':
        L.append("-I")
    elif params['distanceWeighting'] == 'Weight by 1-distance':
        L.append("-F")

    if params['crossValidated'] == True:
        L.append("-X")

    if params['meanSquared'] == True:
        L.append("-E")


    clf = Classifier(classname="weka.classifiers.lazy.IBk")
    return clf


def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])
    L.append("-K")
    L.append(str(params['K']))

    if params['distanceWeighting'] == 'Weight by 1/distance':
        L.append("-I")
    elif params['distanceWeighting'] == 'Weight by 1-distance':
        L.append("-F")

    if params['crossValidated'] == True:
        L.append("-X")

    if params['meanSquared'] == True:
        L.append("-E")


    clf = Classifier(classname="weka.classifiers.lazy.IBk", options=L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result