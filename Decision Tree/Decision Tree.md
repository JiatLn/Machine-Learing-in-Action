# 决策树算法

@[机器学习]

> 工作原理：
用户输入一系列数据，然后给出答案。

- 优点：计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据。
- 缺点：可能会产生过度匹配问题。
- 适用数据类型：数值型和标称型。


## 构造决策树
- 考虑根节点的选取


创建分支的伪代码函数createBranch()如下：
```
检测数据集中的每个子项是否属于同一分类：
If so return 类标签；
Else
    寻找划分数据集的最好特征
    划分数据集
    创建分支节点
        for 每个划分的子集
            (递归)调用函数createBranch并增加返回结果到分支节点中
    return 分支节点
```

本文使用ID3算法划分数据集

### 信息增益

划分数据集的大原则：将无序的数据变得更加有序。

使用信息论度量信息

关于信息增益(information gain)和熵(entropy)

熵定义为信息的期望值

如果待分类的事务可能划分在多个分类中，则符号x<sub>i</sub>的信息定义为

> <a href="https://www.codecogs.com/eqnedit.php?latex=l(x_{i})=-log_{2}p(x_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l(x_{i})=-log_{2}p(x_{i})" title="l(x_{i})=-log_{2}p(x_{i})" /></a>

其中p(x<sub>i</sub>)是选择该分类的概率。

为了计算熵，我们需要计算所有类别所有可能值包含的信息期望值，有如下公式

> <a href="https://www.codecogs.com/eqnedit.php?latex=H=-\sum_{i=1}^{n}p(x_{i})log_{2}p(x_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H=-\sum_{i=1}^{n}p(x_{i})log_{2}p(x_{i})" title="H=-\sum_{i=1}^{n}p(x_{i})log_{2}p(x_{i})" /></a>

其中n是分类的数目。

示例代码：

计算给定数据集的香农熵（用来度量数据集的无序程度）
```
from math import log
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
```

计算出的熵越高，混合的数据也越多。得到熵之后，我们按照获取最大信息增益的方法划分数据集。

### 划分数据集

下面介绍具体学习如何划分数据集以及如何度量信息增益。

按照给定特征划分数据集

```
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            redicedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(featVec[axis+1:])
    return retDataSet
```



