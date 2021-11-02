import random

import scipy.io
import numpy as np
import pandas as pd
from scipy.stats import entropy

if __name__ == '__main__':
    path1 = 'Data for Assignment 3/MindReading/unlabeledMatrix_MindReading1.mat'
    path2 = 'Data for Assignment 3/MindReading/unlabeledLabels_MindReading_1.mat'
    matrix_set = scipy.io.loadmat(path1)['unlabeledMatrix']
    label_set = scipy.io.loadmat(path2)['unlabeledLabels']

    idx = [i for i in range(1000)]
    matrix_df = pd.DataFrame(matrix_set, index=idx)
    label_df = pd.DataFrame(label_set, index=idx)

    random.shuffle(idx)
    # index = np.random.permutation(matrix_df)
    x_unlabeled = matrix_df.reindex(idx)
    y_unlabeled = label_df.reindex(idx)
    x_unlabeled = x_unlabeled.append(x_unlabeled[0:5])
    y_unlabeled = y_unlabeled.append(y_unlabeled[0:5])

    # print(x_unlabeled[-5:], x_unlabeled[0:5])
    # print(y_unlabeled[-5:], y_unlabeled[0:5])
    print(x_unlabeled[0:5])
    print(y_unlabeled[0:5])
    x_unlabeled = x_unlabeled.drop(x_unlabeled.index[[2]])
    y_unlabeled = y_unlabeled.drop(y_unlabeled.index[[2]])
    print(x_unlabeled[0:5])
    print(y_unlabeled[0:5])
