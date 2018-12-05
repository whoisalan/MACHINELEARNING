# Classification  简单示例
from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
clf.predict([[2., 2.]])
# array([1])

# Regression 简单示例
# from sklearn import tree
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])
# array([ 0.5])

# 也可以用sklearn生成可视化的图片(dot文件)
# from sklearn import tree
# tree.export_graphviz(clf,out_file='tree.dot')