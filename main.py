from sklearn.model_selection import train_test_split

from mnist import load_mnist, learn_knn
OPT_K=3  # this num is calced by knn.py


if __name__ == "__main__":
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, shuffle=False)  # the mnist dataset have already shuffled
    knn = learn_knn(X_train, y_train, OPT_K)
