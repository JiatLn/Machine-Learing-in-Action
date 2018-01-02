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
