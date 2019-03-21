from common_defs import *

from weka.classifiers import Classifier


space = {
    'bucket': hp.quniform('ni', 1, 32, 1),
}


def get_params():
    params = sample(space)
    return handle_integers(params)

def get_class(params):
    L = list([])

    L.append("-B")
    L.append(str(params['bucket']))

    clf = Classifier(classname="weka.classifiers.rules.OneR", options=L)
    return clf



def try_params(n_instances, params, train, test, istest):
    n_instances = int(round(n_instances))
    # print "n_instances:", n_instances
    pprint(params)

    L = list([])

    L.append("-B")
    L.append(str(params['bucket']))


    clf = Classifier(classname="weka.classifiers.rules.OneR", options = L)


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, n_instances)


    return result
