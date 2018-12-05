from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
import numpy as np

#n_samples:生成样本的数量
#n_features=2:生成样本的特征数，特征数=n_informative（） + n_redundant + n_repeated
#n_informative：多信息特征的个数
#n_redundant：冗余信息，informative特征的随机线性组合
#n_clusters_per_class ：某一个类别是由几个cluster构成的 
x,y = make_classification(n_samples=1000, n_features=2,n_redundant=0,n_informative=1,n_clusters_per_class=1)

#训练数据和测试数据
x_data_train = x[:800,:]
x_data_test = x[800:,:]
y_data_train = y[:800]
y_data_test = y[800:]

#正例和反例
positive_x1 = [x[i,0] for i in range(1000) if y[i] == 1]
positive_x2 = [x[i,1] for i in range(1000) if y[i] == 1]
negetive_x1 = [x[i,0] for i in range(1000) if y[i] == 0]
negetive_x2 = [x[i,1] for i in range(1000) if y[i] == 0]

# ------------------------------------------------------------------------------------------------------ #

#定义感知机
clf = Perceptron(fit_intercept=False,n_iter=30,shuffle=False)
#使用训练数据进行训练
clf.fit(x_data_train,y_data_train)
#得到训练结果，权重矩阵
print(clf.coef_)
#输出为：[[-0.38478876,4.41537463]]

#超平面的截距，此处输出为：[0.]
print(clf.intercept_)

#利用测试数据进行验证
acc = clf.score(x_data_test,y_data_test)
print(acc)
#得到的输出结果为0.995，这个结果还不错吧。

from matplotlib import pyplot as plt
#画出正例和反例的散点图
plt.scatter(positive_x1,positive_x2,c='red')
plt.scatter(negetive_x1,negetive_x2,c='blue')
#画出超平面（在本例中即是一条直线）
line_x = np.arange(-4,4)
line_y = line_x * (-clf.coef_[0][0] / clf.coef_[0][1]) - clf.intercept_
plt.plot(line_x,line_y)
plt.show()