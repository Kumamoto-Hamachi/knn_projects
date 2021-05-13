from sklearn.model_selection import train_test_split
from mnist import load_mnist
from knn import learn_knn, knn_predict
import pickle
import os
import matplotlib.pyplot as plt

OPT_K = 3


def distinguish_distances(distances, y_pred, y_test):  # correct or wrong
    correct_dis = []
    wrong_dis = []
    for i, (p, t) in enumerate(zip(y_pred, y_test)):
        if p == t:
            correct_dis.append(distances[i])
        else:
            wrong_dis.append(distances[i])
    return correct_dis, wrong_dis


def histogram_with_correct_wrong(correct_dis, wrong_dis, is_show=False):
    fname = "images/histogram_cw.png"
    longest = int(max(max(correct_dis), max(wrong_dis)))
    bins = [i for i in range(0, longest, 100)]
    _, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].set_title("Correct distances")
    axs[1].set_title("Wrong distances")
    axs[0].hist(correct_dis, bins=bins, color="blue")
    axs[1].hist(wrong_dis, bins=bins, color="red")
    plt.savefig(fname)
    if is_show:
        plt.show()


def make_knn_nearest_distacnes(knn, X_test):
    fname = "pickles/distances.pickle"
    if os.path.exists(fname):
        print(f"{fname} exists")  # debug
        neigh_dist_and_indices = pickle.load(open(fname, "rb"))
    else:
        print(f"{fname} doesn't exist")  # debug
        neigh_dist_and_indices = knn.kneighbors(X_test)
        pickle.dump(neigh_dist_and_indices, open(fname, "wb"))
    return neigh_dist_and_indices


if __name__ == "__main__":
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, shuffle=False)  # the mnist dataset have already shuffled
    knn = learn_knn(X_train, y_train, OPT_K)
    neigh_indices = make_knn_nearest_distacnes(knn, X_test)[0]
    distances = [d.mean() for d in neigh_indices]
    y_pred = knn_predict(knn, X_test)
    correct_dis, wrong_dis = distinguish_distances(distances, y_pred, y_test)
    # print("len(correct_dis), len(wrong_dis)", len(correct_dis), len(wrong_dis))  # debug
    histogram_with_correct_wrong(correct_dis, wrong_dis, is_show=True)
