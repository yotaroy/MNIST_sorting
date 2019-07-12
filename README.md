# MNIST sorting

## task
nつの数字の画像が入力され，隣同士の数字を入れ替える行動をして，昇順にソートをして，ソートを終了させる．  
MNISTの数字nつが入力のstateとなり，出力をn-1つのactionとする．
```
s_t = (d_0, d_1, ..., d_n)
a_t = (a_0, a_1, ..., a_(n-1))

s_t: d_0, d_1, ..., d_n はMNISTの画像が1枚ずつ入る
a_n: d_nとd_(n+1)を入れ替える
```

## Requirement
パッケージなど
[requirements.txt](./requirements.txt)  

環境は[env.py](./env.py)に書かれている．ただし，pytorchのtensorを使った学習を想定している．

## Usage
### 使用例

```
import torch
import dqn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model_dict = {'digit':dqn.DQN_DIGIT(), 'mnist':dqn.DQN_MNIST()}

chosen_model = 'mnist'      # 'mnist' or 'digit

learning = dqn.Learning(device, chosen_model, model_dict[chosen_model])
learning.training(EPOCH=1)
```

### [main.py](./main.py)
実行例．実行方法は以下の通り．

```
$ python main.py
```

