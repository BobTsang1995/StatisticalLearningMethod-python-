"""
@File    : Perceptron.py
@Time    : 2020-05-16 
@Author  : BobTsang
@Software: PyCharm
@Email   : bobtsang@bupt.edu.cn
"""
### Perceptron感知机算法的手工实现（鸢尾花数据集）
# mnist_train:80
# mnist_test:20
# score:1.0

import pandas as pd
import numpy as np
import time
from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random


# 手工实现打乱数据，不采用sklearn调用shuffle打乱数据
def Random_number(data_size):
    """
        该函数使用shuffle()打乱一个包含从0到数据集大小的整数列表。因此每次运行程序划分不同，导致结果不同

        改进：
        可使用random设置随机种子，随机一个包含从0到数据集大小的整数列表，保证每次的划分结果相同。

        :param data_size: 数据集大小
        :return: 返回一个列表
    """
    num_set = []
    random.seed(0)
    for i in range(data_size):
        num_set.append(i)
    random.shuffle(num_set)
    return num_set


def Train_test_split(data_set, target_data, size=0.1):
    """
        说明：分割数据集，我这里默认数据集的0.3是测试集

        :param data_set: 数据集
        :param target_data: 标签数据
        :param size: 测试集所占的比率
        :return: 返回训练集数据、训练集标签、训练集数据、训练集标签
    """
    # 计算训练集的数据个数
    train_size = int((1 - size) * len(data_set))

    # 获得数据
    data_index = Random_number(len(data_set))

    # 分割数据集（X表示数据，y表示标签），以返回的index为下标
    x_train = data_set[data_index[:train_size]]

    x_test = data_set[data_index[train_size:]]

    y_train = target_data[data_index[:train_size]]

    y_test = target_data[data_index[train_size:]]
    return x_train, x_test, y_train, y_test


#  感知机模型
def Perceptron(x, y):
    # y = Wx+b
    # 初始化w为全0，长度与每一个样本特征一致
    w = np.zeros(len(x_train[0]))
    # 初始b=0
    b = 0
    # 学习率  也就是我们梯度下降的步长
    lr = 0.01
    # 迭代次数
    iters = 10

    # 返回x的维度（行，列）
    m, n = x.shape
    for iter in range(iters):
        count = 0

        for i, xi in enumerate(x):
            # 计算需要把握好每一个向量的维度，才不容易出错
            # x是80x4的，xi是1x4，y是80x1
            # x是一个样本，y是样本正确分类的结果
            yi = y[i]
            # 在感知机中，误分类点(w*x+b)与y异号，相乘小于0
            # 等于0则点落在超平面上，也不符合要求
            if -yi * (np.dot(w, xi) + b) >= 0:
                # 这些点被归于误差点
                # 根据公式进行梯度下降，使用的是随机梯度下降
                # 每遍历一个误分类样本点，就根据当前样本点进行梯度下降更新参数
                # x是1x4的一维数组，w也是一个一维数组。
                # 所以梯度下降更新w时候，不需要转秩，直接np.dot()，得到点积
                # y是1x1的一维数组,x是1x4的一维数组,取其中的值相乘用np.dot()
                w += lr * np.dot(yi, xi)
                b += lr * yi
                # 误分类样本加一
                count += 1

        # 计算分类正确率
        print(count, m)
        acc = (m - count) / m
        print('Round %d' % (iter), end=' ')
        print('正确率', acc)
    return w, b


def predict(x, y, w, b):
    # x：20x4 数组
    # y：20x1 一维数组
    # 计算根据训练的参数得到的预测y值
    pred = [0] * len(y)
    for i, xi in enumerate(x):
        pred[i] = np.dot(w, xi) + b  # 20x1 列向量

        # f（x）=sign（wx+b）：计算值大于0就=1，小于0就等于-1
        if pred[i] >= 0:
            pred[i] = int(1)
        else:
            pred[i] = int(-1)
    correct = 0
    n = len(pred)
    for i in range(n):
        if pred[i] == y[i]: correct += 1
    score = correct / n
    return score, pred


if __name__ == '__main__':
    iris = datasets.load_iris()
    # 创建一个dataframe表型数据结构
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # 列尾添加新的一列'label', 值为iris.target(Series对象)
    df['label'] = iris.target

    # 重命名列名
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    # 打乱数据
    datas = np.array(df.loc[:99, ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']])
    data, target = datas[:, [0, 1, 2, 3]], datas[:, -1]

    x_train, x_test, y_train, y_test = Train_test_split(data, target)

    # 替换标签把0，1换成感知机公式中的-1，1
    y_train = np.array([1 if i == 1 else -1 for i in y_train])
    # print(x, y)
    # 替换标签把0，1换成感知机公式中的-1，1
    y_test = np.array([1 if i == 1 else -1 for i in y_test])
    # print(x_train, x_test, y_train, y_test)

    w, b = Perceptron(x_train, y_train)
    print(w, b)

    print('原始标签', y_test)

    score, pred = predict(x_test, y_test, w, b)
    print('预测标签', pred)

    print('score', score)

    
