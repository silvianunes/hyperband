from common_defs import *

from weka.classifiers import Classifier

space = {
    'iterations': hp.quniform('it', 1, 10, 1),
    'max_alterations': hp.quniform('ma', 5000, 50000, 1),
    'exponent': hp.uniform('e', 0.2, 5),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)

    L = list([])

    L.append("-I")
    L.append(str(params['iterations']))

    L.append("-M")
    L.append(str(params['max_alterations']))

    L.append("-E")
    L.append(str(params['exponent']))

    clf = Classifier(classname="weka.classifiers.functions.VotedPerceptron", options=L)
    return clf


def try_params(n_instances, params, train, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    L.append("-I")
    L.append(str(params['iterations']))

    L.append("-M")
    L.append(str(params['max_alterations']))

    L.append("-E")
    L.append(str(params['exponent']))

    clf = Classifier(classname="weka.classifiers.functions.VotedPerceptron", options=L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, n_instances)


    return result
