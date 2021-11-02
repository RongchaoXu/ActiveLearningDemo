import pandas as pd
from pandas import DataFrame
import scipy.io
import os
import re


def get_df(matrix_path, label_path, prefix):
    matrix_set = scipy.io.loadmat(matrix_path)[prefix+'Matrix']
    label_set = scipy.io.loadmat(label_path)[prefix+'Labels']
    matrix_df = pd.DataFrame(matrix_set)
    label_df = pd.DataFrame(label_set)
    return matrix_df, label_df


def get_dataset(path):
    mat_pathes = ['', '', '', '', '', '']
    mat_df = []
    mats = os.listdir(path)
    for mat in mats:
        mat_path = os.path.join(path, mat)
        pairs = {
            'trainingMatrix': 0,
            'trainingLabels': 1,
            'testingMatrix': 2,
            'testingLabels': 3,
            'unlabeledMatrix': 4,
            'unlabeledLabels': 5
        }
        for key in pairs.keys():
            if re.match(key, mat):
                mat_pathes[pairs.get(key)] = mat_path
                break

    tmp = get_df(mat_pathes[0], mat_pathes[1], 'training')
    mat_df.append(tmp[0])
    mat_df.append(tmp[1])
    tmp = get_df(mat_pathes[2], mat_pathes[3], 'testing')
    mat_df.append(tmp[0])
    mat_df.append(tmp[1])
    tmp = get_df(mat_pathes[4], mat_pathes[5], 'unlabeled')
    mat_df.append(tmp[0])
    mat_df.append(tmp[1])
    return mat_df


if __name__ == '__main__':
    from utils import plot
    result = get_dataset('Data for Assignment 3/tmp1')
    plot(result)