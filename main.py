from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import os

from mnist import load_mnist
from knn import learn_knn, optimize, knn_predict
from histogram import (make_knn_nearest_distacnes,
        distinguish_distances,
        histogram_with_correct_wrong)
from heatmap_cfm import heatmap_for_cfm
from divide import identify_frequent_combinations
from frequent_error_com import save_test_and_neighs_img
#from util.simple_watch import watch_as_str as w

TRAIN_SIZE = 60000
OPT_K = 3
THRESHOLD_RATE = 0.015

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

def prepare_shuffle_order(seed, orders):
    np.random.seed(seed)
    shuffle_order = np.arange(orders)
    np.random.shuffle(shuffle_order)
    return shuffle_order  # numpy.ndarray

def shuffle_pixcel(dataframe, shuffle_order):
    shuffled_array = dataframe.to_numpy()[:, shuffle_order]
    """ Preparing for the future in case that dataframe is needed
    pixel_list = [f"pixel{i}" for i in range(1, len(shuffle_order))]
    dataframe = pd.DataFrame(data=X_train, columns=pixel_list)
    """
    return shuffled_array  # numpy.ndarray


if __name__ == "__main__":
    # 1. MNIST classfication by using KNN
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, shuffle=False)  # the mnist dataset have already shuffled
    #opt_k = optimize(X_train, y_train)  # opt_k is 3 in this case
    opt_k = OPT_K
    knn = learn_knn(X_train, y_train, opt_k, is_refresh=True)
    score = knn.score(X_test, y_test)
    print("score", score)  # debug

    # 2. Error analysis
    # 2-1. Histogram
    neigh_indices = make_knn_nearest_distacnes(knn, X_test)[0]
    distances = [d.mean() for d in neigh_indices]
    y_pred = knn_predict(knn, X_test)
    correct_dis, wrong_dis = distinguish_distances(distances, y_pred, y_test)
    # print("len(correct_dis), len(wrong_dis)", len(correct_dis), len(wrong_dis))  # debug
    histogram_with_correct_wrong(correct_dis, wrong_dis, is_show=True)

    # 2-2. Heatmapped Confusion matrix
    cfm = confusion_matrix(y_test, y_pred, normalize="true")
    #print("cfm", cfm)  # debug
    classes = knn.classes_
    heatmap_for_cfm(cfm, classes, color_dict, is_show=True)

    # 2-3 Save frequent-error imgs
    frequent_combs = identify_frequent_combinations(y_pred, y_test, THRESHOLD_RATE)
    neigh_idx_list = make_knn_nearest_distacnes(knn, X_test)[1]  # neigh num is three
    for com, test_indices in frequent_combs.items():
        pred_for_test = com[2]
        base_dir = f"images/{com}/"
        os.mkdir(base_dir)
        for test_idx in test_indices:
            idx = test_idx + TRAIN_SIZE  # idx is X's index(before divided X_train from X_test)
            img_dir = f"{base_dir}{str(idx)}/"
            #print("img_dir", img_dir)  # debug
            os.mkdir(img_dir)
            indices = [idx] + list(neigh_idx_list[test_idx])  # neigh_idx_list uses test_idx yet
            #print("indices", indices)  # debug
            save_test_and_neighs_img(indices, img_dir, X, y, pred_for_test)

    # 3 Shulled MNIST classfication by using KNN
    shuffle_order = prepare_shuffle_order(1, 784):  # 28 * 28 = 784
    X_train = shuffle_pixcel(X_train, shuffle_order)
    X_test = shuffle_pixcel(X_test, shuffle_order)
    knn = learn_knn(X_train, y_train, opt_k, is_refresh=True)  # knn must learn shuffled data-set again
    score = knn.score(X_test, y_test)
    print("score", score)  # debug
