from common_defs import *

from weka.attribute_selection import ASSearch

space = {
    'num_attr': hp.quniform('n', 10, 1000, 1),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def get_search(params):
    # pprint(params)

    L = list([])

    L.append("-N")
    L.append(str(params['num_attr']))


    search = ASSearch(classname="weka.attributeSelection.Ranker", options=L)

    return search


