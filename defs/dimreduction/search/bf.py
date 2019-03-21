from common_defs import *

from weka.classifiers import Classifier
from weka.attribute_selection import ASSearch

space = {
    'direction': hp.choice('d', (0, 1, 2)),
    'nodes': hp.quniform('n', 2, 10, 1),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_search(params):
    # pprint(params)

    L = list([])

    L.append("-D")
    L.append(str(params['direction']))

    L.append("-N")
    L.append(str(params['nodes']))

    search = ASSearch(classname="weka.attributeSelection.BestFirst", options=L)

    return search


