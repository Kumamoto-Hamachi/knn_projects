from pprint import pprint as pp
from pprint import pformat as pf
from sklearn.datasets import fetch_openml
import pickle
from sklearn.model_selection import train_test_split

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


if __name__ == "__main__":
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, shuffle=False)  # the mnist dataset have already shuffled
