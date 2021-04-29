from pprint import pprint as pp
from pprint import pformat as pf
from sklearn.datasets import fetch_openml
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

"""
(1)heatmap
why:
goal:
"""


def load_mnist(flg_fetch=False):
    fname = "pickles/mnist.pickle"
    if flg_fetch:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
        with open(fname, "wb") as f:
            pickle.dump((X, y), f)
    else:
        with open(fname, "rb") as f:
            X, y = pickle.load(f)
    print("len(X), len(y)", len(X), len(y))  # debug
    return X, y


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


if __name__ == "__main__":
    #X, y = load_mnist()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, shuffle=False)  # the mnist dataset have already shuffled
    #scores = experiment_knn_score(X_train, y_train)
    #print("scores", scores)  # debug
    #pickle.dump(scores, open("pickles/scores.pickle", "wb"))
    scores = pickle.load(open("pickles/scores.pickle", "rb"))
    print("scores", scores)  # debug
