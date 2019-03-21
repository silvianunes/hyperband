from weka.core.converters import Loader

from weka.core.classes import Random

from weka.core.dataset import Instances
from weka.core.dataset import Instance
from weka.core.dataset import Attribute

import pandas as pd
# from scipy.io import arff
import numpy as np
import scipy
import io
import os
import gzip
import arff

from weka.filters import Filter

class Load_Data():
    def __init__(self, data_dir):
        self.directory = data_dir
        self.data = None
        self.train = None
        self.test = None
        self.valid = None
        self.X = None
        self.y = None
        self.categorical = None

    def return_data(self):
        loader = Loader(classname="weka.core.converters.ArffLoader")

        self.data = loader.load_file(self.directory)

        self.data.class_is_last()

        return self.data

    def return_arff(self):
        filename = self.directory

        with io.open(filename) as fh:
            decoder = arff.ArffDecoder()
            return decoder.decode(fh, encode_nominal=True, return_type=arff.DENSE)

    def split_train_test_valid(self):
        try:
            self.data = self.return_data()
            total_inst = self.data.num_instances
            train_, self.test = self.data.train_test_split(80.0, Random(1))
            self.train, self.valid = train_.train_test_split(75.0, Random(1))

            print('total_inst:  ', total_inst, '| train_inst: ', self.train.num_instances,
                  '| valid_inst: ', self.valid.num_instances, '| test_inst: ', self.test.num_instances)

        except Exception:
             pass

        return self.train, self.valid, self.test

    def split_x_y(self):
        try:
            self.data = self.return_arff()

            self.categorical = [False if type(type_) != list else True
                           for name, type_ in self.data['attributes']]

            attribute_names = [i[0] for i in self.data['attributes']]

            X = np.array(self.data['data'], dtype=np.float32)

            del self.categorical[-1]

            df = pd.DataFrame(X, columns=attribute_names)

            self.X = df[df.columns[:-1]]

            self.y = df[df.columns[-1]]

        except Exception:
             pass

        return self.X, self.y, self.categorical

    def split_x_y_landmarking(self):
        try:

            self.data = self.return_arff()

            # print self.data

            self.categorical = [False if type(type_) != list else True
                           for name, type_ in self.data['attributes']]

            attribute_names = [i[0] for i in self.data['attributes']]

            X = np.array(self.data['data'], dtype=np.float32)

            del self.categorical[-1]

            df = pd.DataFrame(X, columns=attribute_names)

            self.X = df[df.columns[:-1]]

            self.X = self.X.fillna(0)

            self.y = df[df.columns[-1]]

        except Exception:
             pass

        return self.X, self.y, self.categorical
