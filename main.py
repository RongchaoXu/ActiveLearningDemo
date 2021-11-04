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
    parser.add_argument('--type', type=str, default='combine', help='sample strategy:random, entropy, combine')
    parser.add_argument('--solver', type=str, default='newton-cg', help='model solver')
    parser.add_argument('--max_iter', type=int, default=1000, help='max iteration of each training')
    parser.add_argument('--k', type=int, default=10, help='k')
    parser.add_argument('--n', type=int, default=50, help='n')
    parser.add_argument('--plot_type', type=str, default='average'
                        , help='plot single for one case(single) or plot average for entire database(average)')
    args = parser.parse_args()

    datasets = []
    folders = [os.path.join(args.src, f) for f in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, f))]
    if len(folders) == 0:
        datasets.append(args.src)
    else:
        for dataset_path in folders:
            datasets.append(dataset_path)

    result = []
    result1= []
    for dataset_path in datasets:
        dataset = get_dataset(dataset_path)
        if args.type != 'combine':
            tmp_result = active_learning(dataset, get_model(solver=args.solver, max_iter=args.max_iter), strategy=args.type
                                     , k=args.k, n=args.n)
            if len(result) == 0:
                result = tmp_result
            else:
                result = [tmp_result[i]+result[i] for i in range(len(result))]
            if args.plot_type == 'single':
                postfix = args.src.split('/')[-2] + '_' + args.src.split('/')[-1] + args.type + '.png'
                plot(os.path.join(args.dst, postfix), **{args.type:tmp_result})
        else:
            tmp1_result = active_learning(dataset, get_model(solver=args.solver, max_iter=args.max_iter),
                                          strategy='random', k=args.k, n=args.n)
            tmp2_result = active_learning(dataset, get_model(solver=args.solver, max_iter=args.max_iter),
                                          strategy='entropy', k=args.k, n=args.n)
            if len(result) == 0:
                result = tmp1_result
                result1 = tmp2_result
            else:
                result = [tmp1_result[i]+result[i] for i in range(len(result))]
                result1 = [tmp2_result[i]+result1[i] for i in range(len(result))]
            if args.plot_type == 'single':
                postfix = args.src.split('/')[-2] + '_' + args.src.split('/')[-1] + args.type + '.png'
                plot(os.path.join(args.dst, postfix), **{'random': result, 'entropy': result1})

    num = len(folders)
    result = [i/num for i in result]
    result1 = [i/num for i in result1]
    if args.plot_type == 'average':
        postfix = args.src.split('/')[-2] + '_' + args.type + '_' + args.solver + '_' + 'average' + '.png'
        if args.type != 'combine':
            plot(os.path.join(args.dst, postfix), **{args.type: result})
        else:
            plot(os.path.join(args.dst, postfix), **{'random': result, 'entropy': result1})