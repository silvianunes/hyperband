import pandas as pd
import numpy as np
from operator import itemgetter

import cPickle as pickle
import os

# # dir_data = 'C:\Users\silvia\Documents\Silvia-data\\auto-weka datasets\\done\\'
# # dir_results = 'C:\Users\silvia\Dropbox\\ab-cl\\test\\'
# #
# #
# # def main(directory_data, directory_results):
# #
# #     dir_d = os.listdir(directory_data)
# #     dir_r = os.listdir(directory_results)
# #     columns = ['data',  'err', 'time']
# #     columns2 = ['data', 'err', 'time', 'param']
# #     df_average = pd.DataFrame(columns=columns)
# #     df_complete = pd.DataFrame(columns=columns2)
# #
# #     for e in dir_d:
# #         print('NAME:    ', os.path.basename(e))
# #
# #         count = 0
# #         time = 0
# #         err = 0
# #
# #         for i in dir_r:
# #
# #             # print(os.path.basename(e), os.path.basename(i))
# #
# #             if os.path.basename(e) in os.path.basename(i):
# #
# #                 count += 1
# #
# #                 try:
# #                     file = directory_results + os.path.basename(i)
# #
# #                     with open(file, 'rb') as f:
# #                         data = pickle.load(f)
# #
# #                     # print('unsorted:', data)
# #
# #                     data = sorted(data, key=itemgetter('err'))
# #
# #                     print('sorted:', data)
# #
# #                     data = data[0]
# #                     time += data['seconds']
# #                     err += data['err']
# #                     param = data['params']
# #
# #
# #                     df_complete = df_complete.append({'data': [os.path.basename(i)], 'time': data['seconds'], "err": ([data['err']]), "param": param}, ignore_index=True)
# #
# #                 except ValueError:
# #                     pass
# #         if count!=0:
# #             time = time/count
# #             err = err/count
# #             df_average = df_average.append({'data': [os.path.basename(e)], 'time': [time], "err": [err]}, ignore_index=True)
# #
# #             print(df_average)
# #
# #     df_average.to_csv('average-results-autoband-test.csv', sep=',')
# #     df_complete.to_csv('complete-results-autoband-test.csv', sep=',')
# #
# #     return 'finished'
#
#
#
# main(dir_data, dir_results)


import pandas as pd
import numpy as np
from operator import itemgetter

import cPickle as pickle
import os

dir_data = 'C:\Users\silvia\Documents\Silvia-data\\hyperband\\'
dir_results = 'C:\Users\silvia\Desktop\\UB_teste_1\\test\\'


def main(directory_data, directory_results):

    dir_d = os.listdir(directory_data)
    dir_r = os.listdir(directory_results)
    columns = ['data', 'loss', 'err', 'auc', 'time']
    columns2 = ['data', 'loss', 'err', 'auc', 'time', 'param']
    df_average = pd.DataFrame(columns=columns)
    df_complete = pd.DataFrame(columns=columns2)

    for e in dir_d:
        print('DATA:    ', os.path.basename(e))

        count = 0
        loss = 0
        time = 0
        auc = 0
        err = 0

        for i in dir_r:

            if os.path.basename(e) in os.path.basename(i):

                count += 1

                try:
                    file = directory_results + os.path.basename(i)

                    with open(file, 'rb') as f:
                        data = pickle.load(f)

                    print('unsorted:', data)

                    data = sorted(data, key=itemgetter('err'))

                    print('sorted:', data)


                    first = data[0]
                    loss += first['loss']
                    time += first['seconds']
                    auc += first['auc']
                    err += first['err']
                    param = first['params']


                    df_complete = df_complete.append({'data': [os.path.basename(i)], 'loss': [first['loss']], 'time': [first['seconds']], "auc": [first['auc']], "err": [first['err']], "param": [first['params']]}, ignore_index=True)

                except ValueError:
                    pass
        if count!=0:
            loss = loss/count
            time = time/count
            auc = auc/count
            err = err/count
            df_average = df_average.append({'data': [os.path.basename(e)], 'loss': [loss], 'time': [time], "auc": [auc], "err": [err]}, ignore_index=True)

            print(df_average)

    df_average.to_csv('average-results-ub-teste.csv', sep=',')
    df_complete.to_csv('complete-results-ub-teste.csv', sep=',')

    return 'finished'



main(dir_data, dir_results)




