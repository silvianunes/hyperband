import defs.dimreduction.search.bf as bf
import defs.dimreduction.search.gs as gs
from weka.attribute_selection import ASEvaluation
from weka.classifiers import Classifier

import defs.dimreduction.search.rk as rk
from common_defs import *

space = {
    'missing_merge': hp.choice('mm', (True, False)),
    # 'search': hp.choice('search', ('GreedyStepwise', 'BestFirst', 'Ranker')),
}

def get_params():
    params = sample(space)
    return handle_integers(params)

def get_evaluator(params, base):
    pprint(params)

    L = list()

    if params['missing_merge'] == True:
        L.append("-M")


    # if params['search'] == 'GreedyStepwise':
    #     param_search = gs.get_params()
    #     search = gs.get_search(param_search)
    # elif params['search'] == 'BestFirst':
    #     param_search = bf.get_params()
    #     search = bf.get_search(param_search)
    # elif params['search'] == 'Ranker':
    param_search = rk.get_params()
    search = rk.get_search(param_search)

    # search = ASSearch(classname="weka.attributeSelection."+params['search'])
    evaluator = ASEvaluation(classname="weka.attributeSelection.GainRatioAttributeEval", options=L)

    clf = Classifier(classname="weka.classifiers.meta.AttributeSelectedClassifier")

    clf.set_property("evaluator", evaluator.jobject)
    clf.set_property("search", search.jobject)
    clf.set_property("base", base.jobject)


    return clf

#
def try_params(n_instances, params, base, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list()
    if params['missing_merge'] == True:
        L.append("-M")


    if params['search'] == 'GreedyStepwise':
        param_search = gs.get_params()
        search = gs.get_search(param_search)
    elif params['search'] == 'BestFirst':
        param_search = bf.get_params()
        search = bf.get_search(param_search)
    elif params['search'] == 'Ranker':
        param_search = rk.get_params()
        search = rk.get_search(param_search)

    # search = ASSearch(classname="weka.attributeSelection."+params['search'])
    evaluator = ASEvaluation(classname="weka.attributeSelection.GainRatioAttributeEval", options=L)

    clf = Classifier(classname="weka.classifiers.meta.AttributeSelectedClassifier")

    clf.set_property("evaluator", evaluator.jobject)
    clf.set_property("search", search.jobject)
    clf.set_property("base", base.jobject)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result
