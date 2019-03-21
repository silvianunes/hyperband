from common_defs import *

from weka.classifiers import Classifier


space = {

}


def get_params():
    params = {}
    return params


def get_class(params):
    # pprint(params)
    clf = Classifier(classname="weka.classifiers.bayes.NaiveBayesMultinomial")
    return clf


def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)
    
    clf = Classifier(classname="weka.classifiers.bayes.NaiveBayesMultinomial")


    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result