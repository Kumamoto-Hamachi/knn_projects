from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

from mnist import load_mnist
from knn import learn_knn, knn_predict
from util.simple_watch import watch_as_str as w
from divide import identify_frequent_combinations

OPT_K = 3  # This num is calced by knn.py
THRESHOLD_RATE = 0.015  # This num represents the rate of each labels


""" Frequent error combs (check the heatmap img)
7-1 0.02
4-9 0.019
8-3 0.016
"""


def save_X_image(X, indices, base_dir):
    for i in indices:
        filename = base_dir + str(i)
        plt.imshow(X.to_numpy()[i].reshape(28, 28), cmap=plt.cm.gray_r)
        plt.savefig(filename)


if __name__ == "__main__":
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, shuffle=False)  # the mnist dataset have already shuffled
    knn = learn_knn(X_train, y_train, OPT_K)
    y_pred = knn_predict(X_train, y_train, OPT_K, X_test)
    frequent_combs = identify_frequent_combinations(y_pred, y_test, THRESHOLD_RATE)
    print("frequent_combs", frequent_combs)  # debug
    print(w("prepared,,,"))
    for com, indices in frequent_combs.items():
        base_dir = f"images/{com}/"
        os.mkdir(base_dir)
        save_X_image(X_test, indices, base_dir)
        print(w(base_dir))
