from common_defs import *

from weka.classifiers import Classifier

space = {
    'neighbours': hp.choice('n', (0, hp.choice('n', (-1, 30, 60, 90, 120)))),
    'weighting': hp.choice('w', (0, hp.choice('w', (0, 1, 2, 3, 4)))),
    'search': hp.choice('s', ("weka.core.neighboursearch.LinearNNSearch")),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])

    L.append("-K")
    L.append(str(params['neighbours']))

    L.append("-U")
    L.append(str(params['weighting']))

    L.append("-A")
    L.append("weka.core.neighboursearch.LinearNNSearch")

    clf = Classifier(classname="weka.classifiers.lazy.LWL", options=L)
    return clf


def try_params(n_instances, params, train, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    L.append("-K")
    L.append(str(params['neighbours']))

    L.append("-U")
    L.append(str(params['weighting']))

    L.append("-A")
    L.append(str(params['search']))


    clf = Classifier(classname="weka.classifiers.lazy.LWL", options=L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, n_instances)


    return result
