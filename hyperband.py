import numpy as np

from random import random
from math import log, ceil
from time import time, ctime

# loading data
# from load_data import load

from pprint import pprint

class Hyperband:

    def __init__(self, get_params_function, try_params_function, train, valid, test):

        self.get_params = get_params_function
        self.try_params = try_params_function

        self.train = train
        self.valid = valid
        self.test = test

        # self.time_limit = time_limit

        self.max_inst = round(train.num_instances) # maximum instances per configuration

        self.n_max = np.maximum(9, self.max_inst/700)   #maximum configurations
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

            # n random configurations
            T = [self.get_params() for i in range(n)]

            # for i in range((s + 1) - int(skip_last)): # changed from s + 1
            for i in range(s+1):  # changed from s + 1

                # Run each of the n configs for <instances>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_instances = r * self.eta ** (i)

                print("\n*** {} configurations x {:.1f} instances each".format(n_configs, n_instances))

                # val_losses = []
                # early_stops = []

                for t in T:

                    self.counter += 1
                    print("\n{} | {} | highest  auc so far: {:.4f} (run {})\n".format(self.counter, ctime(),
                                                                                      self.max_auc, self.best_counter))

                    start_time = time()

                    if dry_run:
                        result = {'loss': random(), 'auc': random(), 'acc': random(), 'err': random()}
                    else:

                        result = self.try_params(n_instances, t, self.train, self.valid, self.test, False) # <--


                    assert(type(result) == dict)
                    assert('auc' in result)

                    seconds = int(round(time() - start_time))
                    print("\n{} seconds.".format(seconds))

                    auc = result['auc']


                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if auc > self.max_auc:
                        self.max_auc = auc
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['instances'] = n_instances

                    self.results.append(result)

                # select a number of best configurations for the next loop
                # filter out early stops, if any
                remain_conf = int(n_configs / self.eta)
                T = self.top_k(self.results, remain_conf)
        return self.results

    def top_k(self, results, remain_conf):

        T = [r['params'] for r in sorted(results, key=lambda x: x['auc'], reverse=True)[:remain_conf]]

        return T

    # it is called just one time

    def tests(self, results):

        for r in sorted(results, key=lambda x: x['err'])[:5]:

            params = r['params']

            start_time = time()

            test_result = self.try_params(self.max_inst, params, self.train, self.valid, self.test, True)

            print(test_result)

            assert (type(test_result) == dict)
            assert ('auc' in test_result)

            seconds = int(round(time() - start_time))
            print("\n{} seconds.".format(seconds))

            # loss = test_result['loss']

            test_result['counter'] = r['counter']
            test_result['seconds'] = seconds
            test_result['params'] = r['params']

            self.test_results.append(test_result)

        return self.test_results

