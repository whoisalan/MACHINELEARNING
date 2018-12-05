import numpy
import matplotlib.pyplot as plot

#建立一个100数据的测试集
nPoints = 100

#x的取值范围：-0.5～+0.5的nPoints等分
xPlot = [-0.5+1/nPoints*i for i in range(nPoints + 1)]

#y值：在x的取值上加一定的随机值或者叫噪音数据
#设置随机数算法生成数据时的开始值，保证随机生成的数值一致
numpy.random.seed(1)
##随机生成宽度为0.1的标准正态分布的数值
##上面的设置是为了保证numpy.random这步生成的数据一致
y = [s + numpy.random.normal(scale=0.1) for s in xPlot]


#离差平方和列表
sumSSE = []
for i in range(1, len(xPlot)):
    #以xPlot[i]为界，分成左侧数据和右侧数据
    lhList = list(xPlot[0:i])
    rhList = list(xPlot[i:len(xPlot)])

    #计算每侧的平均值
    lhAvg = sum(lhList) / len(lhList)
    rhAvg = sum(rhList) / len(rhList)

    #计算每侧的离差平方和
    lhSse = sum([(s - lhAvg) * (s - lhAvg) for s in lhList])
    rhSse = sum([(s - rhAvg) * (s - rhAvg) for s in rhList])

    #统计总的离差平方和，即误差和

    sumSSE.append(lhSse + rhSse)

##找到最小的误差和
minSse = min(sumSSE)
##产生最小误差和时对应的数据索引
idxMin = sumSSE.index(minSse)
##打印切割点数据及切割点位置
print("切割点位置："+str(idxMin)) ##49
print("切割点数据："+str(xPlot[idxMin]))##-0.010000000000000009

##绘制离差平方和随切割点变化而变化的曲线
plot.plot(range(1, len(xPlot)), sumSSE)
plot.xlabel('Split Point Index')
plot.ylabel('Sum Squared Error')
plot.show()