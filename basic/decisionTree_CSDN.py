from math import log
import operator

def calcShannonEnt(dataSet):  # 计算数据的熵(entropy)
    numEntries=len(dataSet)  # 数据条数
    labelCounts={}
    
    for featVec in dataSet:  # 对于np.array的遍历是按行来的
        currentLabel=featVec[-1] # 每行数据的最后一个字（类别）
        if currentLabel not in labelCounts.keys(): # 如果这一类已经在字典里出现过了
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1  # 统计有多少个类以及每个类的数量
    
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries # 计算单个类的熵值，古典概率
        shannonEnt-=prob*log(prob,2) # 累加每个类的熵值
    return shannonEnt

def createDataSet1():    # 创造示例数据
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    labels = ['头发','声音']  #两个特征
    return dataSet,labels

def splitDataSet(dataSet,axis,value): # 按某个特征分类后的数据
    retDataSet=[]
    for featVec in dataSet:    # np.array的遍历是按行遍历的
        if featVec[axis]==value:
            reducedFeatVec =featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):  # 选择最优的分类特征
    numFeatures = len(dataSet[0])-1   # 特征的个数
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = 0   # 信息增益
    bestFeature = -1   # 最优的特征
    for i in range(numFeatures):  # 按照特征遍历
        featList = [example[i] for example in dataSet]  # 一条数据
        uniqueVals = set(featList)  # 一条数据的所有类别(某一个特征的所有类别)
        newEntropy = 0
        
        for value in uniqueVals: # 对这个特征下的每一个类别遍历计算
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet)  # 按特征分类后的熵，是每一种类别加起来的
        
        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
        
        if (infoGain>bestInfoGain):   # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):    #按分类后类别数量排序，比如：最后分类为2男1女，则判定为男；
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]  # 类别：男或女

    if classList.count(classList[0])==len(classList): # 训练集中标签全是男/女
        return classList[0]
    
    if len(dataSet[0])==1: # 训练集只有一个
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet) #选择最优特征
    bestFeatLabel=labels[bestFeat]  # 上面返回的是序号
    myTree={bestFeatLabel:{}} #分类结果以字典形式保存
    
    del(labels[bestFeat]) # 她的label是features的意思....搞我
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues) # 找到这个feature下的所有类别
    
    for value in uniqueVals: # 根据每一种类别分别建树
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


if __name__=='__main__':
    dataSet, labels=createDataSet1()  # 创造示列数据
    print(createTree(dataSet, labels))  # 输出决策树模型结果
