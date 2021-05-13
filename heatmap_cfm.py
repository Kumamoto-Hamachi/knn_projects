from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from mnist import load_mnist
from knn import learn_knn, knn_predict
from util.simple_watch import watch_as_str as w
import os
import pickle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

OPT_K = 3  # this num is calced by knn.py

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


def prepare_disp(cfm, classes, cmap):
    _, ax = plt.subplots(figsize=(13, 8))
    """ same process
    fig = plt.figure(figsize=(13, 8))
    ax = fig.subplots()
    #"""
    disp = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=classes)
    disp.plot(cmap=cmap, ax=ax)


def heatmap_for_cfm(cfm, classes, color_dict, is_show=False):
    fname = "images/heatmap.png"
    cmap = LinearSegmentedColormap("custom_cmap", color_dict)
    prepare_disp(cfm, classes, cmap)
    plt.savefig(fname)
    if is_show:
        plt.show()


if __name__ == "__main__":
    w()
    X, y = load_mnist()
    print(w(cmt="load mnist"))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, shuffle=False)  # the mnist dataset have already shuffled
    print(w(cmt="train split"))
    knn = learn_knn(X_train, y_train, OPT_K)
    print(w(cmt="knn learn"))
    y_pred = knn_predict(X_train, y_train, OPT_K, X_test)
    print(w(cmt="knn predict"))
    cfm = confusion_matrix(y_test, y_pred, normalize="true")
    print(w(cmt="make cfm"))
    print("cfm", cfm)  # debug
    classes = knn.classes_
    heatmap_for_cfm(cfm, classes, color_dict, is_show=True)
