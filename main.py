import os
from train import get_model
from train import active_learning
from dataset import get_dataset
from utils import plot
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Active learning')
    parser.add_argument('--src', type=str, help='dataset path')
    parser.add_argument('--dst', type=str, default='imgs', help='destination path')
    parser.add_argument('--type', type=str, default='random', help='sample strategy')
    parser.add_argument('--solver', type=str, default='newton-cg', help='model solver')
    parser.add_argument('--max_iter', type=int, default=1000, help='max iteration of each training')
    parser.add_argument('--k', type=int, default=10, help='k')
    parser.add_argument('--n', type=int, default=50, help='n')
    args = parser.parse_args()

    dataset = get_dataset(args.src)
    result = active_learning(dataset, get_model(solver=args.solver, max_iter=args.max_iter), strategy=args.type
                             , k=args.k, n=args.n)
    postfix = args.src.split('/')[-2] + '_' + args.src.split('/')[-1] + args.type + '.png'
    plot(result, os.path.join(args.dst, postfix))