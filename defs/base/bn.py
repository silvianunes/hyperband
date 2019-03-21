from common_defs import *


from weka.classifiers import Classifier


space = {
    'useAT': hp.choice('u', (True, False)),
    'search': hp.choice('search', ('weka.classifiers.bayes.net.search.local.K2', 'weka.classifiers.bayes.net.search.local.HillClimber',
                                   'weka.classifiers.bayes.net.search.local.LAGDHillClimber', 'weka.classifiers.bayes.net.search.local.SimulatedAnnealing',
                                    'weka.classifiers.bayes.net.search.local.TabuSearch', 'weka.classifiers.bayes.net.search.local.TAN')),
}


def get_params():
    params = sample(space)
    return handle_integers(params)

def get_class(params):
    # pprint(params)

    L = list([])

    if params['useAT'] == False:
        L.append("-D")

    L.append("-Q")
    L.append(params['search'])

    clf = Classifier(classname="weka.classifiers.bayes.BayesNet", options = L)

    return clf


def try_params(n_instances, params, train, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    if params['useAT'] == False:
        L.append("-D")

    L.append("-Q")
    L.append(params['search'])

    clf = Classifier(classname="weka.classifiers.bayes.BayesNet", options = L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, n_instances)


    return result
