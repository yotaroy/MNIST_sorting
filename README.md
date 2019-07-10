# MNIST sorting

## task
4つの数字の画像が入力され，隣同士の数字を入れ替える行動をして，昇順にソートをして，ソートを終了させる．  
MNISTの数字4つが入力のstateとなり，出力を4つのactionとする．
```
s_t = (n_0, n_1, n_2, n_3)
a_t = (a_0, a_1, a_2, a_3)

s_t: n_0, n_1, n_2, n_3 はMNISTの画像が1枚ずつ入る
a_0: n_0とn_1を入れ替える
a_1: n_1とn_2を入れ替える
a_2: n_2とn_3を入れ替える
a_3: ソートを終わらせる(報酬をもらう)
```

## Requirement
パッケージなど
[requirements.txt](./requirements.txt)

## Usage
