from common_defs import *

from weka.classifiers import Classifier
from weka.attribute_selection import ASSearch

space = {
    'conservation': hp.choice('c', (True, False)),
    'backward': hp.choice('b', (True, False)),
    'ranked': hp.choice('r', (True, False)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_search(params):
    # pprint(params)

    L = list([])

    if params['conservation'] == False:
        L.append("-C")

    if params['backward'] == False:
        L.append("-B")

    if params['ranked'] == False:
        L.append("-R")

    search = ASSearch(classname="weka.attributeSelection.GreedyStepwise", options=L)

    return search


