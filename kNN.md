k-近邻算法，属于分类算法

工作原理：
存在一个样本数据集合（训练样本集），样本集中每个数据都存在标签（对应的分类）。
输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，
然后算法提取样本集中特征最相似数据（最近邻）的分类标签。

一般只取前k个最相似的数据，出现次数最多的分类作为新数据的分类。
通常k是不大于20的整数。


伪代码：
对未知类别属性的数据集中的每个点依次执行一下操作：
- 计算已知类别属性的数据集中的点与当前之间的距离；
- 按照距离递增次序排序；
- 选取与当前点距离最小的k个点；
- 确定前k个点所在的类别的出现频率；
- 返回前k个点出现频率最高的类别作为当前点的预测分类。

Python 3.X 示例代码：
```
import random

import numpy as np
import matplotlib.pyplot as plt


def knn_classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    distances = calcDistance(inX, dataSet)
    pair = sorted(zip(distances, labels))
    voteLabel = []
    for i in range(k):
        voteLabel.append(pair[i][1])
    return max(voteLabel, key=voteLabel.count)


# 计算欧拉距离，返回inX与dataSet中各个点的距离列表
def calcDistance(inX, dataSet):
    # inX, dataSet均为numpy.ndarray
    distances = []
    for data in dataSet:
        distances.append(np.linalg.norm(inX - data))
    return distances


# 载入数据集
def loadDataSet(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        dataSet = []
        for line in lines:
            dataSet.append(line.strip().split(','))
        random.shuffle(dataSet)
        group = []
        labels = []
        for data in dataSet:
            labels.append(data.pop())
            group.append(data)
        return np.array(group, dtype=float), np.array(labels)


def showDataSet(dataSet, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:, 0], dataSet[:, 3], 15.0*labels, 15.0*labels)
    plt.show()


# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # print(normDataSet[0:5])
    return normDataSet


# 分类结果数字化
def text2num(labels):
    newLabels = []
    for label in labels:
        if label == 'Iris-setosa':
            newLabels.append(1)
        elif label == 'Iris-versicolor':
            newLabels.append(2)
        elif label == 'Iris-virginica':
            newLabels.append(3)
    return np.array(newLabels)


def main():
    inX = np.array([7., 3., 5., 1.])
    group, labels = loadDataSet(r'C:\Users\Administrator\Desktop\iris.data')
    group = autoNorm(group)
    labels = text2num(labels)
    count = len(labels)
    trainSet, trainLabel = group[0:int(count*0.8)], labels[0:int(count*0.8)]
    testSet, testLabel = group[int(count*0.8):], labels[int(count*0.8):]
    print('已知分类样本个数：', len(trainLabel))
    print('待分类样本个数：', len(testLabel))
    errorCount = 0
    for i in range(len(testLabel)):
        result = knn_classify(testSet[i], trainSet, trainLabel, 10)
        print('预测分类结果为{0}, 实际分类为{1}'.format(result, testLabel[i]))
        if result != testLabel[i]:
            errorCount += 1
    print('the total error rate is:', errorCount / 50)



if __name__ == '__main__':
    main()

```
函数说明：

- knn_classify(inX, dataSet, labels, k)
本算法的核心
参数说明：
inX：待分类的特征向量
dataSet：训练样本集
labels：训练样本对应的分类标签
k：选择最近邻的数目




- loadDataSet(path)
加载数据集，在实验中，采用的是 IRIS数据集
返回两个numpy的array对象，分别是特征集合以及对应的分类标签

- autoNorm(dataSet)
特征归一化，将特征值转化为0到1区间内的值
newValue = (oldValue - min) / (max - min)

- def showDataSet(dataSet, labels)
展示数据集，在这里只是将数据集的第0和第3个特征作为x/y轴，绘制成散点图
从结果可以看出，不同分类之间的数据呈现出区别

- text2num(labels)
将文本标签转化为数字
Iris-setosa     对应类别1
Iris-versicolor 对应类别2
Iris-virginica  对应类别3
