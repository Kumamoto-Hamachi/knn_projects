from sklearn.datasets import fetch_openml
import pickle
import os
import make_dir


def load_mnist():
    base_dir = "pickles"
    make_dir(base_dir)
    fname = f"{base_dir}/mnist.pickle"
    if os.path.exists(fname):
        print(f"{fname} exists")  # debug
        with open(fname, "rb") as f:
            X, y = pickle.load(f)
    else:
        print(f"{fname} doesn't exist")  # debug
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
        with open(fname, "wb") as f:
            pickle.dump((X, y), f)
    print("len(X), len(y)", len(X), len(y))  # debug
    return X, y
