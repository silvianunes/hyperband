from common_defs import *


from weka.classifiers import Classifier


space = {
    'unpruned': hp.choice('up', (True, False)),
    'collapseTree': hp.choice('ct', (True, False)),
    'confidenceFactor': hp.uniform('cf', 0, 1),
    'minNumObj': hp.quniform('mno', 0, 64, 1),
    'binarySplits': hp.choice('bs', (True, False)),
    'subtreeRaising': hp.choice('st', (True, False)),
    'useLaplace': hp.choice('ul', (True, False)),
    'useMDL': hp.choice('um', (True, False)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_class(params):
    # pprint(params)
    L = list([])

    if params['unpruned'] == True:
        L.append("-U")

    if params['collapseTree'] == True:
        L.append("-O")

    if params['unpruned'] == False:
        L.append("-C")
        L.append(str(params['confidenceFactor']))

    L.append("-M")
    L.append(str(params['minNumObj']))

    if params['binarySplits'] == True:
        L.append("-B")

    if params['subtreeRaising'] == True and params['unpruned'] == False:
        L.append("-S")

    if params['useLaplace'] == True:
        L.append("-A")

    if params['useMDL'] == False:
        L.append("-J")

    clf = Classifier(classname="weka.classifiers.trees.J48", options=L)

    return clf


def try_params(n_instances, params, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list([])

    if params['unpruned'] == True:
        L.append("-U")

    if params['collapseTree'] == True:
        L.append("-O")

    if params['unpruned'] == False:
        L.append("-C")
        L.append(str(params['confidenceFactor']))

    L.append("-M")
    L.append(str(params['minNumObj']))

    if params['binarySplits'] == True:
        L.append("-B")

    if params['subtreeRaising'] == True and params['unpruned'] == False:
        L.append("-S")

    if params['useLaplace'] == True:
        L.append("-A")

    if params['useMDL'] == False:
        L.append("-J")

    clf = Classifier(classname="weka.classifiers.trees.J48", options=L)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result
