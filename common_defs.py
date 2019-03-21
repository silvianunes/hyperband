import numpy as np

from math import log, sqrt
from time import time
from pprint import pprint

from weka.classifiers import Evaluation
from weka.core.classes import Random

try:
    from hyperopt import hp
    from hyperopt.pyll.stochastic import sample
except ImportError:
    print("In order to achieve operational capability, this programme requires hyperopt to be installed (pip install hyperopt), unless you make get_params() use something else.")


#

# handle floats which should be integers
# works with flat params
def handle_integers(params):
    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v

    return new_params


###


def train_and_eval_weka_classifier(clf, train, valid, n_instances):

    total_train_inst = train.num_instances

    percentage = (n_instances*100)/total_train_inst

    if percentage == 100:
        opt = train
    else:
        opt, residual = train.train_test_split(percentage, Random(1))

    # opt, residual = train.train_test_split(percentage, Random(1))

    print('total_train_inst:    ', total_train_inst, '| percentage:    ', percentage, '| used_inst:     ', opt.num_instances)

    clf.build_classifier(opt)

    evl = Evaluation(opt)
    evl.test_model(clf, valid)
    # evl.crossvalidate_model(clf, opt, 10, Random(1))

    acc = evl.percent_correct
    auc = evl.weighted_area_under_roc
    err = evl.error_rate
    log = evl.sf_mean_scheme_entropy

    print("# validating  | loss: {:.2}, accuracy: {:.4}, AUC: {:.2}, error: {:.2}".format(log, acc, auc, err))

    return {'loss': log, 'accuracy': acc, 'auc': auc, 'err': err}


def test_weka_classifier(clf, train, test):

    clf.build_classifier(train)

    evl = Evaluation(train)
    evl.test_model(clf, test)

    acc = evl.percent_correct
    auc = evl.weighted_area_under_roc
    err = evl.error_rate
    log = evl.sf_mean_scheme_entropy


    print("# testing  | loss: {:.2}, accuracy: {:.4}, AUC: {:.2}, error: {:.2}".format(log, acc, auc, err))

    return {'loss': log, 'accuracy': acc, 'auc': auc, 'err': err}