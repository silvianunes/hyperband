from numpy.random import choice


def get_indexed_workflow(row, column, index):
    return {'fs': column, 'learner': row, 'index': index}


def get_space(weights):
    rows = list(weights.index.get_values())
    columns = list(weights.head())

    indexed_workflows = []
    probabilities = []
    workflows = []
    index = 0

    for i in range(weights.shape[1]):
        for j in range(weights.shape[0]):
            prob = weights.iloc[j, i]
            iw = get_indexed_workflow(row=rows[j], column=columns[i], index=index)
            indexed_workflows.append(iw)
            probabilities.append(prob)
            workflows.append(iw['index'])
            index+=1
    space = {'indexed_workflows': indexed_workflows, 'probabilities': probabilities, 'workflows': workflows}
    return space


def get_names(indexed_workflow):

    params = {}
    if indexed_workflow['learner'] == 'AB':
        params['learner'] = 'meta.ada'
    elif indexed_workflow['learner'] == 'BN':
        params['learner'] = 'base.bn'
    elif indexed_workflow['learner'] == 'DS':
        params['learner'] = 'base.dec'
    elif indexed_workflow['learner'] == 'DT':
        params['learner'] = 'base.dt'
    elif indexed_workflow['learner'] == 'HT':
        params['learner'] = 'base.ht'
    elif indexed_workflow['learner'] == 'Ibk':
        params['learner'] = 'base.knn'
    elif indexed_workflow['learner'] == 'J48':
        params['learner'] = 'base.j48'
    elif indexed_workflow['learner'] == 'Jrip':
        params['learner'] = 'base.jr'
    elif indexed_workflow['learner'] == 'LMT':
        params['learner'] = 'base.knn'
    elif indexed_workflow['learner'] == 'Log':
        params['learner'] = 'base.log'
    elif indexed_workflow['learner'] == 'LB':
        params['learner'] = 'meta.lb'
    elif indexed_workflow['learner'] == 'MLP':
        params['learner'] = 'base.mlp'
    elif indexed_workflow['learner'] == 'NB':
        params['learner'] = 'base.nb'
    elif indexed_workflow['learner'] == 'ONE':
        params['learner'] = 'base.oneR'
    elif indexed_workflow['learner'] == 'PART':
        params['learner'] = 'base.part'
    elif indexed_workflow['learner'] == 'RC':
        params['learner'] = 'meta.rc'
    elif indexed_workflow['learner'] == 'RF':
        params['learner'] = 'base.rf'
    elif indexed_workflow['learner'] == 'RS':
        params['learner'] = 'meta.rs'
    elif indexed_workflow['learner'] == 'RT':
        params['learner'] = 'base.rt'
    elif indexed_workflow['learner'] == 'REP':
        params['learner'] = 'base.rep'
    elif indexed_workflow['learner'] == 'SGD':
        params['learner'] = 'base.knn'
    elif indexed_workflow['learner'] == 'SL':
        params['learner'] = 'base.sl'
    elif indexed_workflow['learner'] == 'SVM':
        params['learner'] = 'base.svm'
    elif indexed_workflow['learner'] == 'STK':
        params['learner'] = 'meta.stc'
    elif indexed_workflow['learner'] == 'VOTE':
        params['learner'] = 'meta.vote'
    elif indexed_workflow['learner'] == 'VP':
        params['learner'] = 'base.vp'
    elif indexed_workflow['learner'] == 'ZERO':
        params['learner'] = 'base.zeroR'


    if indexed_workflow['fs'] == 'CAE':
        params['featureSelection'] = 'cae'
    elif indexed_workflow['fs'] == 'CFS':
        params['featureSelection'] = 'cfs'
    elif indexed_workflow['fs'] == 'GR':
        params['featureSelection'] = 'gr'
    elif indexed_workflow['fs'] == 'IG':
        params['featureSelection'] = 'ig'
    elif indexed_workflow['fs'] == 'ONE':
        params['featureSelection'] = 'oae'
    elif indexed_workflow['fs'] == 'RL':
        params['featureSelection'] = 'relief'
    elif indexed_workflow['fs'] == 'SU':
        params['featureSelection'] = 'su'
    elif indexed_workflow['fs'] == 'WP':
        params['featureSelection'] = 'wr'
    elif indexed_workflow['fs'] == 'SINGLE':
        params['featureSelection'] = None

    return params


def weighted_sample(space):
    indexed_workflows = space['indexed_workflows']
    probabilities = space['probabilities']
    workflows = space['workflows']

    wf = choice(workflows, 1, p=probabilities)

    params = get_names(indexed_workflows[int(wf)])

    return params
