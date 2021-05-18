# KNN Project
このドキュメントは未完成です。

# Requirements
```
scikit_learn==0.24.1
numpy==1.20.2
matplotlib==3.4.1
Pillow==8.2.0
```

# 前提知識
## K nearest neighbors
K nearest neighbors(以下、KNNと呼ぶ)はテストデータが入力されたときにそれに近いk個のトレーニングデータを取り、それらのラベルの多数決を採ることで、テストデータのラベルを予測するモデルである。
## MNIST
<img width="915" alt="mnist_samples" src="https://user-images.githubusercontent.com/51290155/118697387-31f28480-b84a-11eb-8639-22deab9bef91.png">
[MNIST](http://yann.lecun.com/exdb/mnist/)(Modified National Institute of Standards)は0~9の手書き数字の画像データセットだ。データセットには60,000枚のトレーニングデータと10,000枚のテストデータが含まれている。
MNIST分類を行うために元となるデータは[OpenMLのmnist_784](https://www.openml.org/d/554)から取得した。

# 1. KNNを用いたMNIST分類
MNISTのテストデータセットのそれぞれの手書き数字画像データのラベル(0~9)をKNNを用いて予測した。
このラベル予測のことを本稿では以後、「MNIST分類」と呼ぶことにする。

## KNNモデルのパラメータ kの選択
KNNでラベル予測のために取得するトレーニングデータの個数kk(パラメータ)を選択する。
TODO:バリデーションセットのこと記述

## モデルの学習と予測
バリデーションセットを分割する前のトレーニングデータをkk=3でKNNに学習させる。
さらにその学習済みのモデルを用いて、テストデータをMNIST分類させた。

結果、ラベル推測の正答率は97.05%となった。

# 2. MNIST分類のエラー分析
MNIST分類の際に生じたエラー(誤ったラベル予測)の分析のために(1)外れ値の可能性検証(2)頻出するエラーのを行う。

そのために(1)ヒストグラムによる正解したデータとエラーになったデータの分布比較(2)混合行列(Confusion Matrix)による確認を行なう。


## ヒストグラムの作成と分布の確認
テストデータと3つのneighborの特徴量それぞれのユークリッド距離の計算結果の平均(以下、ユークリッド距離と呼ぶ)を横軸とする。横軸の最小値は0、最大値は上記のデータセットのユークリッド距離平均の中で最大のものを採用する。また縦軸はそれぞれのユークリッド距離ごとのデータ数とした。

先ほどMNIST分類したテストデータのうち正解した物とエラーになった物を分け、それぞれを用いて上記の定義に従ったヒストグラムを作成する。
![histogram_cw](https://user-images.githubusercontent.com/51290155/118698484-6a469280-b84b-11eb-92de-f66e78b80abb.png)

上記から正解したデータの分布は1100~1200に、エラーとなったデータ分布には1600~1700に、山(データ数が最も多い箇所)が出来ている。また誤りデータの分布は正解したデータの分布に比べ、全体的に見ても右側に偏っていることがわかる。

以上からデータセットの正誤によって分布の仕方に差があり、誤りデータの一部に外れ値があるとわかった。

## 混合行列の作成と頻出するエラーの組み合わせの確認
ラベル推測を誤ったデータにはどんな特徴があるのかを探るために、混同行列(Confusion Matrix)を用いて確認する。またデータ数の大小を視覚的に確認するために、混合行列をヒートマップ化した。

![heatmap](https://user-images.githubusercontent.com/51290155/118698600-906c3280-b84b-11eb-93c0-f30a64975715.png)
「正解ラベル-予測ラベル」のうち上記の混合行列で0.015以上となるのうち「7-1」、「4-9」、「8-3」の組み合わせを頻出エラーと呼ぶことにする。
次は頻出エラー及びそのneighborの画像3枚を確認する。
![4-9](https://user-images.githubusercontent.com/51290155/118698917-ec36bb80-b84b-11eb-94d8-baa08f85a135.png)
![7-1](https://user-images.githubusercontent.com/51290155/118698934-efca4280-b84b-11eb-8e43-34da5b035b32.png)
![8-3](https://user-images.githubusercontent.com/51290155/118698937-f062d900-b84b-11eb-963a-beb1769ab4b8.png)
上記の画像を筆者の目視で確認した結果(1)「人間の目で見ても数字を判断出来ないテストデータ」もある一方で(2)「充分に数値判断出来るテストデータ」もあることがわかった。

# 考察
TODO
一般的に画像データ(ここでバラバラの画像データ出す)は人間の目で見て(正解の)判断がつくものであるべきだ。
<img width="709" alt="row_shuffle" src="https://user-images.githubusercontent.com/51290155/118699135-2607c200-b84c-11eb-8172-243ac22808c2.png">
※右は左側の画像のピクセルをシャッフルしたもの

しかし、本記事で行なってきたモデルの学習/予測のアルゴリズムではピクセルの光度を(全て同じ規則で)シャッフルしてもスコアの精度に影響が出ることはない。
※実際に一定の規則に基づいてシャッフルしたデータを学習/予測に用いた結果、正答率はシャッフルする前と同じく97.05%となった。

# 参考文献
[Machine Learning: A Probabilistic Perspective (Adaptive Computation and Machine Learning series) Kevin P. Murphy](https://www.amazon.co.jp/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020)
