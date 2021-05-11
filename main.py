from sklearn.model_selection import train_test_split

from mnist import load_mnist
from knn import learn_knn, knn_predict
from util.simple_watch import watch_as_str as w
import pickle
import matplotlib.pyplot as plt

OPT_K = 3
"""
7-1 2.0%
4-9 1.9%
8-3 1.6%
"""

#TODO 7-1, 4-9, 8-3それぞれのindicesをだせ => まさかハードコーディングしないだろうな？
def divide_specified_indices(y_pred, y_test):  # starts at 60000
    correct_indices = []
    wrong_indices = []
    for i, (pred, ans) in enumerate(zip(y_pred, y_test)):
        if pred == ans:
            correct_indices.append(i)
        else:
            wrong_indices.append(i)
    return correct_indices, wrong_indices


# TODO
def save_X_image(X, indices):  # 関数で実験しろよ。
    filename = "images/"
    for i in indices:
        #TODO to_numpyで書き直すこと
        plt.imshow(X_test.values[i].reshape(28, 28), cmap="gray")
        plt.savefig(filename)
        break


if __name__ == "__main__":
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, shuffle=False)  # the mnist dataset have already shuffled
    knn = learn_knn(X_train, y_train, OPT_K)
    y_pred = knn_predict(X_test)
    correct_indices, wrong_indices = divide_specified_indices(y_pred, y_test)
    #print("X_test", X_test.values[0].reshape(28, 28))  # debug
    # TEST start
    plt.imshow(X_test.values[0].reshape(28, 28), cmap="gray")  #TODO ハードコーディング直せ
    plt.show()
    # TEST end
