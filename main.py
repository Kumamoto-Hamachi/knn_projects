from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from mnist import load_mnist
from knn import learn_knn
from util.simple_watch import watch_as_str as w
import os
import pickle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

OPT_K=3  # this num is calced by knn.py

color_dict = {
     'red':   [
         (0.0,  1.0, 1.0),
         (0.01,  0.9, 0.9),
         (0.02,  1.0, 1.0),
         (0.5,  1.0, 1.0),
         (1.0,  0.0, 0.0)
         ],
     'green': [
         (0.0,  1.0, 1.0),
         (0.005,  1.0, 1.0),
         (0.015,  0.0, 0.0),
         (1.0,  0.0, 0.0)
         ],
     'blue':  [
         (0.0,  1.0, 1.0),
         (0.005,  1.0, 1.0),
         (0.015,  0.0, 0.0),
         (1.0,  1.0, 1.0)
         ]
     }

def knn_predict(X_test):
    fname = "pickles/y_pred.pickle"
    if os.path.exists(fname):
        print(f"{fname} exists")  # debug
        y_pred = pickle.load(open(fname, "rb"))
    else:
        print(f"{fname} doesn't exist")  # debug
        y_pred = knn.predict(X_test)
        pickle.dump(y_pred, open(fname, "wb"))
    return y_pred



if __name__ == "__main__":
    w()
    X, y = load_mnist()
    print(w(cmt="load mnist"))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, shuffle=False)  # the mnist dataset have already shuffled
    print(w(cmt="train split"))
    knn = learn_knn(X_train, y_train, OPT_K)
    print(w(cmt="knn learn"))
    y_pred = knn_predict(X_test)
    print(w(cmt="knn predict"))
    cfm = confusion_matrix(y_test, y_pred, normalize="true")
    print(w(cmt="make cfm"))
    print("cfm", cfm)  # debug
    cmap = LinearSegmentedColormap("custom_cmap", color_dict)
    _, ax = plt.subplots(figsize=(13, 8))
    """ same process
    fig = plt.figure(figsize=(8, 6))
    ax = fig.subplots()
    #"""
    disp = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=knn.classes_)
    disp.plot(cmap=cmap, ax=ax)
    plt.show()
