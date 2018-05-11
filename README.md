### 计算新闻相似度

#### 详细信息

给定一些新闻（训练集`train.csv`)，训练模型，计算给定`test.csv`中每一条新闻在训练集中最相似的20条新闻。

#### 数据集下载地址

[train_data.csv](https://drive.google.com/open?id=1CwkRBhYZ4hCQFVxxVTKXBd-CyF9Q_CMn)
[test.csv](https://drive.google.com/open?id=1YtkQcrytys8_brck4xF68T0BYsdFskeW)

* 注:如果打不开，翻墙试试

#### 方法

实现了四种方法:

* LCS
* jacarrd
* tfidf
* tfidf+word2vec

使用各种方式的方法: 

* `python main.py --method=[method name] ...`

#### 补充

直接使用训练数据训练word2vec效果不一定会很好。后来有人使用`BM25`算法效果还不错。
