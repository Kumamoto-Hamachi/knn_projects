from pprint import pprint as pp
from pprint import pformat as pf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os

from util.util import argmax_for_dict
from mnist import load_mnist

"""
(1)heatmap
why:
goal:
"""


def experiment_knn_score(X, y):
    scores = {}
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, shuffle=False)
    for neigh_num in range(1, 6):
        knn = KNeighborsClassifier(n_neighbors=neigh_num)
        knn.fit(X_train, y_train)
        score = knn.score(X_val, y_val)
        print("score", score)  # debug
        scores[neigh_num] = score
    return scores


def optimize(X, y):
    fname = "pickles/scores.pickle"
    if os.path.exists(fname):
        print(f"{fname} exists")  # debug
        scores = pickle.load(open(fname, "rb"))
    else:
        print(f"{fname} doesn't exist")  # debug
        scores = experiment_knn_score(X, y)
    opt_k, max_score = argmax_for_dict(scores)
    print("opt_k", opt_k)  # debug
    return opt_k


def learn_knn(X, y, k):
    fname = "pickles/knn.pickle"
    if os.path.exists(fname):
        print(f"{fname} exists")  # debug
        knn = pickle.load(open(fname, "rb"))
    else:
        print(f"{fname} doesn't exist")  # debug
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        pickle.dump(knn, open(fname, "wb"))
    return knn



if __name__ == "__main__":
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, shuffle=False)  # the mnist dataset have already shuffled
    #scores = experiment_knn_score(X_train, y_train)
    #print("scores", scores)  # debug
    #pickle.dump(scores, open("pickles/scores.pickle", "wb"))
    #scores = pickle.load(open("pickles/scores.pickle", "rb"))
    #print("scores", scores)  # debug
    opt_k = optimize(X_train, y_train)
    knn = learn_knn(X_train, y_train, opt_k)
    score = knn.score(X_test, y_test)
    print("score", score)  # debug
