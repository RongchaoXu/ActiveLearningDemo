import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import copyfile


def plot(result, saving_path='a.png'):
    arr = np.asarray(result, dtype=np.float32)
    plt.plot(arr)
    plt.savefig(saving_path)


def dataset_split(src, dst):
    mats = os.listdir(src)
    for mat in mats:
        num = mat.strip('.mat')[-1]
        dir_path = os.path.join(dst, 'case'+num)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        copyfile(os.path.join(src, mat), os.path.join(dir_path, mat))


if __name__ == '__main__':
    dataset_split('Data for Assignment 3/MindReading', 'datasets/MindReading')
    # dataset_split('Data for Assignment 3/MMI', 'datasets/MMI')