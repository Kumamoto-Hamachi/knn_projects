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


def save_test_and_neighs_imgs(test_idx, neighs_idx, img_dir, X_train, X_test):
    filename = img_dir + str(test_idx) + ".png"
    plt.imshow(X_test.to_numpy()[test_idx].reshape(28, 28), cmap=plt.cm.gray_r)
    plt.savefig(filename)
    for ni in neighs_idx:
        filename = img_dir + str(ni) + ".png"
        plt.imshow(X_train.to_numpy()[ni].reshape(28, 28), cmap=plt.cm.gray_r)
        plt.savefig(filename)


def save_combined_test_and_neighs_img(test_idx, neighs_idx, img_dir, y_train, y_test, pred):
    img_list = [None] * 4
    img_order = 0
    filename = img_dir + str(test_idx) + ".png"
    label = y_test.to_numpy()[test_idx]
    img_list[img_order] = prepare_img(filename, test_idx, label, pred=pred)
    img_order += 1
    for ni in neighs_idx:
        filename = img_dir + str(ni) + ".png"
        img_list[img_order] = prepare_img(filename, ni, label)
        img_order += 1
    concatenated_img = cocat_img_list_horizontally(img_list)
    concatenated_img.save(f"{img_dir}/{str(test_idx)}_quad.png")


def prepare_img(filename, order, label, pred=None):
    img = Image.open(filename)
    img = img.crop((100, 20, 540, 460))
    draw = ImageDraw.Draw(img)
    font1 = ImageFont.truetype("/System/Library/Fonts/ヒラギノ明朝 ProN.ttc", 16)
    font2 = ImageFont.truetype("/System/Library/Fonts/ヒラギノ明朝 ProN.ttc", 24)
    draw.multiline_text((330, 10), f"Order:{order}", fill=(0, 0, 0), font=font1)
    draw.multiline_text((60, 45), f"Label:{label}", fill=(0, 0, 0), font=font2)
    if pred:
        draw.multiline_text((60, 70), f"Pred:{pred}", fill=(0, 0, 0), font=font2)
    return img


def cocat_img_list_horizontally(img_list):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, shuffle=False)  # the mnist dataset have already shuffled
    knn = learn_knn(X_train, y_train, OPT_K)
    y_pred = knn_predict(knn, X_test)
    frequent_combs = identify_frequent_combinations(y_pred, y_test, THRESHOLD_RATE)
    print("frequent_combs", frequent_combs)  # debug
    neigh_idx_list = make_knn_nearest_distacnes(knn, X_test)[1]  # neigh num is three
    print("neigh_idx_list", neigh_idx_list)  # debug
    for com, indices in frequent_combs.items():
        pred = com[2]
        base_dir = f"images/{com}/"
        os.mkdir(base_dir)
        for i in indices:
            img_dir = f"{base_dir}{str(i)}/"
            os.mkdir(img_dir)
            save_test_and_neighs_imgs(i, neigh_idx_list[i], img_dir, X_train, X_test)
            save_combined_test_and_neighs_img(i, neigh_idx_list[i], img_dir, y_train, y_test, pred)


"""
(1)data-set
テストデータ
トレーニングデータ

(2)process
KNNの学習-
KNNによるラベル予測-
予測ラベルと実際のラベル比較-
頻出ラベルに対応するneighbor(トレーニングデータ)のindexを取得-
頻出ラベル、そのneighbor、それぞれに対応する画像の保存
上記の4枚の画像の連結(順番通りになるように、それぞれのlabelとorderも載せる)


テストデータ保存
ネイバーデータ保存
1枚にまとめて(テストにはpredをつける,全てにlabelとorderをつける)保存
pandas勉強しよう
"""
