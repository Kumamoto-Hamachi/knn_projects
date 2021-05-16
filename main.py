from mnist import load_mnist
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math
import os

from divide import identify_frequent_combinations
from knn import learn_knn, knn_predict
from histogram import make_knn_nearest_distacnes

OPT_K = 3
THRESHOLD_RATE = 0.015
TRAIN_SIZE = 60000


def save_test_and_neighs_img(indices, img_dir, X, y, pred_for_test):
    img_list = [None] * len(indices)
    test_and_neighs_img_name = f"{str(indices[0])}_quad"  # first idx is test and
    for order, idx in enumerate(indices):
        filename = img_dir + str(idx) + ".png"
        plt.imshow(X.to_numpy()[idx].reshape(28, 28), cmap=plt.cm.gray_r)
        plt.savefig(filename)
        pred = pred_for_test if order == 0 else None  # first is test so shold add pred-label
        img_list[order] = prepare_img(idx, filename, y, pred=pred)
    concatenated_img = concat_imgs_horizontally(img_list)
    concatenated_img.save(f"{img_dir}/{test_and_neighs_img_name}.png")


def prepare_img(idx, filename, y, pred=None):
    label = y[idx]
    img = Image.open(filename)
    img = img.crop((100, 20, 540, 460))
    draw = ImageDraw.Draw(img)
    font1 = ImageFont.truetype("/System/Library/Fonts/ヒラギノ明朝 ProN.ttc", 16)
    font2 = ImageFont.truetype("/System/Library/Fonts/ヒラギノ明朝 ProN.ttc", 24)
    draw.multiline_text((330, 10), f"Index:{idx}", fill=(0, 0, 0), font=font1)
    draw.multiline_text((60, 45), f"Label:{label}", fill=(0, 0, 0), font=font2)
    if pred:
        draw.multiline_text((60, 70), f"Pred:{pred}", fill=(0, 0, 0), font=font2)
    return img


def concat_imgs_horizontally(img_list):
    min_width = math.inf
    max_height = 0
    sum_width = 0
    for img in img_list:
        min_width = img.width if min_width > img.width else min_width
        max_height = img.height if img.height > max_height else max_height
        sum_width += img.width
    img_foundation = Image.new('RGB', (sum_width, max_height))
    x = 0
    for img in img_list:
        img_foundation.paste(img, (x, 0))
        x += img.width
    return img_foundation


if __name__ == "__main__":
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, shuffle=False)  # the mnist dataset have already shuffled
    knn = learn_knn(X_train, y_train, OPT_K)
    y_pred = knn_predict(knn, X_test)
    frequent_combs = identify_frequent_combinations(y_pred, y_test, THRESHOLD_RATE)
    #print("frequent_combs", frequent_combs)  # debug
    neigh_idx_list = make_knn_nearest_distacnes(knn, X_test)[1]  # neigh num is three
    #print("neigh_idx_list", neigh_idx_list)  # debug
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
