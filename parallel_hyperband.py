import numpy as np

from random import random
from math import log, ceil
from time import time, ctime

# loading data
# from load_data import load

from pprint import pprint

class Hyperband:

    def __init__(self, get_workflow_function, get_params_function, try_params_function, train, valid, test):

        self.get_workflow = get_workflow_function
        self.get_params = get_params_function
        self.try_params = try_params_function

        self.train = train
        self.test = test
        self.valid = valid

        # self.time_limit = time_limit

        self.max_inst = round(train.num_instances) # maximum instances per configuration

        self.n_max = np.maximum(12, self.max_inst/700)   #maximum configurations
        self.eta = 3			# defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        # self.s_max = int(self.logeta(self.max_inst))
        self.s_max = int(self.logeta(self.n_max))    # with n_max
        self.B = (self.s_max + 1) * self.max_inst

        self.results = []       # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.min_error = np.inf
        self.max_auc = 0
        self.best_counter = -1
        self.test_results = [] #list of dicts


    # can be called multiple times

    def run(self, skip_last=0, dry_run=False):
        print('max ins', self.max_inst)
        print('n_max', self.n_max)
        print('eta', self.eta)
        print('s_max', self.s_max)
        print('B', self.B)

        for s in reversed(range(self.s_max + 1)):
            print('s', s)
            # initial number of configurations
            n = int(ceil(self.B / self.max_inst / (s + 1) * self.eta ** s))

            print('n', n)

            # initial number of instances per config
            r = self.max_inst * self.eta ** (-s)

            print('r', r)

            results_shw = self.successive_halving_workflows(num_conf=n, num_resources=r, s=s, dry_run=dry_run)
            for i in results_shw:
                self.results.append(i)

        return self.results


    def successive_halving_workflows(self, num_conf, num_resources, s, dry_run):
        # Run each of the n configs for <instances>
        # and keep best (n_configs / eta) configurations

        results_wf = []

        # n random configurations
        wf = [self.get_workflow() for i in range(num_conf)]

        # for i in range((s + 1) - int(skip_last)): # changed from s + 1
        for i in range(s + 1):

            n_configs = num_conf * self.eta ** (-i)
            n_instances = num_resources * self.eta ** (i)

            print("\n*** {} configurations x {:.1f} instances each".format(n_configs, n_instances))

            for t in wf:

                self.counter += 1
                print("\n{} | {} | highest auc so far: {:.4f} (run {})\n".format(self.counter, ctime(),
                                                                                  self.max_auc, self.best_counter))
                start_time = time()

                result_shp, all_results = self.successive_halving_hyperparameters(num_conf, num_resources, s, dry_run, t)

                assert (type(result_shp) == dict)
                assert ('auc' in result_shp)

                seconds = int(round(time() - start_time))
                print("\n{} seconds.".format(seconds))

                auc = result_shp['auc']

                # keeping track of the best result so far (for display only)
                # could do it be checking results each time, but hey
                if auc > self.max_auc:
                    self.max_auc = auc
                    self.best_counter = self.counter

                result_shp['params_complete'] = result_shp['params']
                result_shp['seconds'] = seconds
                result_shp['params'] = t
                result_shp['instances'] = n_instances

                results_wf.append(result_shp)

            remain_conf = int(n_configs / self.eta)
            wf = self.top_k(results_wf, remain_conf)

        return results_wf

    def successive_halving_hyperparameters(self, num_conf, num_resources, s, dry_run, wf_param):
        # Run each of the n configs for <instances>
        # and keep best (n_configs / eta) configurations

        results_shp = []

        # n random configurations
        conf = [self.get_params(wf_param) for i in range(num_conf)]

        # for i in range((s + 1) - int(skip_last)): # changed from s + 1
        for i in range(s + 1):

            n_configs = num_conf * self.eta ** (-i)
            n_instances = num_resources * self.eta ** (i)

            print("\n*** {} configurations x {:.1f} instances each".format(n_configs, n_instances))

            for t in conf:

                self.counter += 1
                print("\n{} | {} | highest auc so far: {:.4f} (run {})\n".format(self.counter, ctime(),
                                                                                  self.max_auc,
                                                                                  self.best_counter))
                start_time = time()

                if dry_run:
                    result_tp = {'loss': random(), 'auc': random(), 'acc': random(), 'err': random()}
                else:

                    result_tp = self.try_params(n_instances, t, self.train, self.valid, self.test, False)  # <--

                assert (type(result_tp) == dict)
                assert ('auc' in result_tp)

                seconds = int(round(time() - start_time))
                print("\n{} seconds.".format(seconds))

                auc = result_tp['auc']

                # keeping track of the best result so far (for display only)
                # could do it be checking results each time, but hey
                if auc > self.max_auc:
                    self.max_auc = auc
                    self.best_counter = self.counter

                result_tp['counter'] = self.counter
                result_tp['seconds'] = seconds
                result_tp['params'] = t
                result_tp['instances'] = n_instances

                results_shp.append(result_tp)

            # select a number of best configurations for the next loop
            remain_conf = int(n_configs / self.eta)
            # conf = conf[0:int(n_configs / self.eta)]
            conf = self.top_k(results_shp, remain_conf)

        best_result = sorted(results_shp, key=lambda x: x['auc'], reverse=True)[0]
        return best_result, results_shp

    def top_k(self, results, remain_conf):

        T = [r['params'] for r in sorted(results, key=lambda x: x['auc'], reverse=True)[:remain_conf]]

        return T

