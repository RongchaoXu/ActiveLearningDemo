import random
from sklearn.linear_model import LogisticRegression
from dataset import get_dataset
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import numpy as np


def get_model(solver='newton-cg', max_iter=1000):
    return LogisticRegression(solver=solver, max_iter=max_iter)


def train_model(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train)
    pred_labels = model.predict(x_test)
    acc = accuracy_score(y_test, pred_labels)
    return model, acc


def active_learning(dataset, model, strategy='random', k=10, n=50):
    x_train = dataset[0]
    y_train = dataset[1]
    x_test = dataset[2]
    y_test = dataset[3]

    x_unlabeled = dataset[4]
    y_unlabeled = dataset[5]
    index = [i for i in range(len(x_unlabeled))]
    x_unlabeled = x_unlabeled.reindex(index)
    y_unlabeled = y_unlabeled.reindex(index)

    results = []
    start = 0
    model, acc = train_model(x_train, y_train, x_test, y_test, model)
    results.append(acc)
    for i in range(n):
        if strategy == 'random':
            new_x = x_unlabeled[start:start+k]
            new_y = y_unlabeled[start:start+k]
            start += k
            x_train = x_train.append(new_x)
            y_train = y_train.append(new_y)
        elif strategy == 'entropy':
            prob = model.predict_proba(x_unlabeled)
            entropies = entropy(prob, axis=1, base=2)
            indices = sorted(range(len(entropies)), key=lambda j: entropies[j])[-1*k:]
            for index in indices:
                x_train = x_train.append(x_unlabeled.iloc[[index]])
                y_train = y_train.append(y_unlabeled.iloc[[index]])
                x_unlabeled.drop(x_unlabeled.index[[index]])
                y_unlabeled.drop(y_unlabeled.index[[index]])

        model, acc = train_model(x_train, y_train, x_test, y_test, model)
        print(acc)
        results.append(acc)
    return results


if __name__ == '__main__':
    from utils import plot
    dataset = get_dataset('datasets/MMI/case1')
    # train_model(datasets[0], datasets[1], datasets[2], datasets[3], get_model())
    result = active_learning(dataset, get_model(), strategy='entropy')
    plot(result)