from common_defs import *

from weka.classifiers import Classifier


space = {
    'blend': hp.quniform('b', 1, 100, 1),
    'entropic': hp.choice('e', (True, False)),
    'missing': hp.choice('m', ('a', 'd', 'm', 'n')),
}


def get_params():
    params = sample(space)
    return handle_integers(params)

def get_class(params):
    L = list([])

    L.append("-B")
    L.append(str(params['blend']))

    if params['entropic'] == True:
        L.append("-E")

    L.append("-M")
    L.append(params['missing'])

    clf = Classifier(classname="weka.classifiers.lazy.KStar", options=L)

    return clf



def try_params(n_instances, params, train, test, istest):

    n_instances = int(round(n_instances))
    # print "n_instances:", n_instances
    pprint(params)

    L = list([])

    L.append("-B")
    L.append(str(params['blend']))

    if params['entropic'] == True:
        L.append("-E")

    L.append("-M")
    L.append(params['missing'])


    clf = Classifier(classname="weka.classifiers.lazy.KStar", options=L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, n_instances)


    return result