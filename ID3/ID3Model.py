from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calShannonEnt(dataSet):
    numEntries = len(dataSet)
    # 为所有的分类类目创建字典
    labelCount = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    # 计算香农熵
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key] / numEntries)
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 定义按照某个特征进行划分的函数splitDataSet
# 输入三个变量（待划分的数据集，特征，分类值）
def splitDataSet(dataSet, axis, value):
    retDatSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDatSet.append(reduceFeatVec)
    return retDatSet


# 定义按照最大信息增益划分数据的函数
def chooseBestFeatureToSpilt(dateSet):
    numFeature = len(dataSet[0]) - 1  # 特征值个数
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeature):
        featList = [number[i] for number in dataSet]  # 得到某个特征下所有值（某列）
        uniqualVals = set(featList)  # set无重复的属性特征值
        newEntropy = 0
        for value in uniqualVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataSet)  # 对各子集香农熵求和
        infoGain = baseEntropy - newEntropy
        # 最大信息增益
        if infoGain > bestInfoGain:
            baseEntropy = infoGain
            bestFeature = i
    return bestFeature


# 投票表决代码
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别相同，停止划分
    if classList.count(classList[-1]) == len(classList):
        return classList[-1]
    # 长度为1，返回出现次数最多的类别
    if len(classList) == 1:
        return majorityCnt(classList)
    # 按照信息增益最高选取分类特征属性
    bestFeat = chooseBestFeatureToSpilt(dataSet)  # 返回分类的特征序号
    bestFeatLable = labels[bestFeat]  # 该特征的label
    myTree = {bestFeatLable: {}}  # 构建树的字典
    del (labels[bestFeat])  # 从labels的list中删除该label
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)  # 所有关于该特征的非重复的值（除掉用过的）

    for value in uniqueVals:
        subLabels = labels[:]  # 子集合（除了该特征的所有的特征）
        myTree[bestFeatLable][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 决策树运用于分类
def classify(inputTree, featLables, testVec):
    # [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    firstStr = list(inputTree.keys())[0]  # 获取树的第一个特征属性
    secondDict = inputTree[firstStr]  # 树的分支，子集合Dict
    featIndex = featLables.index(firstStr)  # 获取决策树第一层在featLables中的位置
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLables, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    # print(dataSet)
    # print(labels)
    # print(calShannonEnt(dataSet))
    # print(splitDataSet(dataSet, 0, 0))
    # print(splitDataSet(dataSet, 0, 1))
    # print(chooseBestFeatureToSpilt(dataSet))
    # print(createTree(dataSet, labels))
    trees = createTree(dataSet, labels)
    dataSet, labels = createDataSet()
    print(classify(trees, labels, [1, 1]))
