from pprint import pprint as pp
from pprint import pformat as pf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os

from util.util import argmax_for_dict
from mnist import load_mnist


def experiment_knn_score(X, y):  #TODO: add pickle
    scores = {}
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, shuffle=False)
    for neigh_num in range(1, 6):
        knn = KNeighborsClassifier(n_neighbors=neigh_num)
        knn.fit(X_train, y_train)
        score = knn.score(X_val, y_val)
        #print("score", score)  # debug
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


def learn_knn(X, y, k, is_refresh=False):
    fname = "pickles/knn.pickle"
    if os.path.exists(fname) and not is_refresh:
        print(f"{fname} exists")  # debug
        knn = pickle.load(open(fname, "rb"))
    else:
        print(f"{fname} doesn't exist")  # debug
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        pickle.dump(knn, open(fname, "wb"))
    return knn


def knn_predict(learned_knn, X_test):
    fname = "pickles/y_pred.pickle"
    if os.path.exists(fname):
        print(f"{fname} exists")  # debug
        y_pred = pickle.load(open(fname, "rb"))
    else:
        print(f"{fname} doesn't exist")  # debug
        y_pred = learned_knn.predict(X_test)
        pickle.dump(y_pred, open(fname, "wb"))
    return y_pred
