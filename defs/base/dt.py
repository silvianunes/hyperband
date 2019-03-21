from common_defs import *


from weka.classifiers import Classifier


space = {
    'evaluation': hp.choice('ev', ('acc', 'rmse', 'mae', 'auc')),
    'useNN': hp.choice('nn', (True, False)),
    'search': hp.choice('sc', ('BestFirst', 'GreedyStepwise', 'Ranker')),
    'cv': hp.choice('cv', (1, 2, 3, 4)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])
    L.append("-E")
    L.append(str(params['evaluation']))

    if params['useNN'] == True:
        L.append("-I")

    L.append("-S")
    L.append(str(params['search']))

    L.append("-X")
    L.append(str(params['cv']))


    clf = Classifier(classname="weka.classifiers.rules.DecisionTable", options=L)
    return clf


def try_params(n_instances, params, train, valid, test, istest):

    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])
    L.append("-E")
    L.append(str(params['evaluation']))

    if params['useNN'] == True:
        L.append("-I")

    L.append("-S")
    L.append(str(params['search']))

    L.append("-X")
    L.append(str(params['cv']))


    clf = Classifier(classname="weka.classifiers.rules.DecisionTable", options=L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result
