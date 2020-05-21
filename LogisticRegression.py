"""
@File    : LogisticRegression.py
@Time    : 2020-05-20 
@Author  : BobTsang
@Software: PyCharm
@Email   : bobtsang@bupt.edu.cn
"""
import numpy as np
import pandas as pd
from sklearn import datasets

# 逻辑回归和感知机还有线性回归之间的关系
# 按我的理解来说，感知机就是在线性回归的基础上套了一层sign函数逻辑回归
# 而逻辑回归是在线性回归的基础上套了一层sigmold函数
# 这里我只实现了逻辑回归解决二分类问题，多分类问题，要么借助k个分类器进行分类，要么换成softmax进行分类
# 之前的感知机算法用的是np.array，即数组来做的，这次可以尝试用np.matrix,即矩阵来做。
# 划分数据集,将数据集划分为训练集和测试集
def Train_test_split(data, size=0.3):

    # 打乱索引
    index = [i for i in range(len(data))]
    np.random.seed(1)
    np.random.shuffle(index)

    # 以8：2的比例划分为训练集和测试集
    train = data.loc[index[:int(len(data) * (1-size))]]
    test = data.loc[index[int(len(data) * (1-size)):]]

    return train, test
  
# 定义sigmold函数
def sigmold(x):
    z = 1 / (1 + np.exp(-x))

    return z

# 定义标准化函数，可以更快收敛
# z-score标准化(zero-mean normalization)
# 经过处理的数据符合标准正态分布，即均值为0，标准差为1
def Standard(xMat):
    inMat = xMat.copy()
    # 求每一列的均值
    # axis=0意味着取这列的每一行出来求均值，最终得到所有的列的均值
    inMeans = np.mean(inMat, axis=0)
    # 求每一列的方差
    inVar = np.std(inMat, axis=0)
    # 每一个特征属性标准化
    inMat = (inMat - inMeans) / inVar
    # 将DataFrame格式转化为矩阵
    inMat = np.mat(inMat)
    return inMat
  
# 定义极大似然估计和交叉熵
def MLE_CrossEntropy(data, target, weights, bias):
    """
        针对于所有的样本数据，计算（负的）对数极大似然估计，也叫做交叉熵
        这个值越小越好
        data: 训练数据（特征向量）， 大小为m * n
        target: 训练数据（标签），一维的向量，长度为n
        weights: 模型的参数， 一维的向量，长度为n
        bias: 模型的偏移量，标量
    """
    # 首先按照标签来提取0和1的下标，由公式可以得到P(Y|X)=P(y=1)^y x P(y=0)^1-y
    # 当y取0的时候前面一项为1，当y取1时后面一项取1，且P(y=0) = 1 - P(y=1) ,P(y=1) = 1/(1+exp^-(w.T*x+b))
    # 取对数的目的是为了归一化，把乘法变成加法    
    idx0, idx1 = np.where(target == 0), np.where(target == 1)
    P0_sum = 0
    P1_sum = 1
    # 根据极大似然估计公式知道
    for i in range(len(idx0)):
        P0_sum += np.log(1 - sigmold(np.dot(data[i], weights) + bias))
    for j in range(len(idx1)):
        P1_sum += np.log(sigmold(np.dot(data[j], weights) + bias))
    # 交叉熵
    crossentropy = -(P0_sum + P1_sum)
    return crossentropy
  
# 实现mini_batch下的梯度下降逻辑回归模型
def MBGD_LogisticRegression(data, lr = 0.001, Epochs = 1000):
    """
        基于梯度下降法实现逻辑回归模型
        xMat: 训练数据（特征向量）， 大小为M * N,一定要注意格式！
        yMat: 训练数据（标签），一维的向量，长度为N
        Epochs: 梯度下降法的迭代次数
        lr: 学习率，步长
    """
    # 将DataFrame格式的数据转化为np.matrix格式    
    yMat = np.mat(data.iloc[:, -1].values).T
    xMat = Standard(data.iloc[:, :-1])
    # print(xMat.shape[1]) # 测试
    # print(xMat) # 测试
    # 返回xMat的维度（行，列）
    m, n = xMat.shape
    # print(m, n)
    # 初始化weights为全0，长度与每一个样本特征一致,有几个特征，定义几个权重值
    # 注意：不是一维数组，而是1x4的矩阵
    weights = np.zeros((n, 1))
    # 初始化权重为0
    bias = 0
    for iter in range(Epochs):
        # batch_size为随机采样大小，数据集比较小，这里有个问题，epochs在神经网络中是完完整整数据集遍历一次的循环次数，即数据总量/Batch_Size = iteration
        # 我这里没有那么做。
        batch_size = 4
        # 随机的索引值
        idx = np.random.choice(xMat.shape[0], batch_size)
        batch_x = xMat[idx]
        batch_y = yMat[idx]
        # 计算预测值与实际值之间的误差
        # 由公式可知W‘ = -Sigma(1->N)[yi-σ(w.T*x+b)]*xi,把负号提进去得到[]里面的值就是下面所说的error        
        error = sigmold(np.dot(batch_x, weights) + bias) - batch_y

        # 对于w, b的梯度计算
        grad_w = np.matmul(batch_x.T, error)
        grad_b = np.sum(error)

        # 对权重和偏置的梯度更新
        weights = weights - lr * grad_w
        bias = bias - lr * grad_b

        if iter % 100 == 0:
            print(iter, MLE_CrossEntropy(xMat, yMat, weights, bias))
    return weights, bias
  

def predict(data, target, weights, bias):
    '''
    data:array形式
    target:array形式的一维数组
    weights:matrix形式的矩阵
    bias:标量
    '''  
    pred = [0] * len(target)
    # 将矩阵数组展平为一维数组
    weights = weights.A.flatten()
    # print(weights) # 测试
    # print(weights.shape)
    for i, xi in enumerate(data):
        print(xi)
        pred[i] = np.dot(weights, xi) + bias

        # 决策边界为y = 0.5
        # f（x）=sigmold（W.T*x+b）：计算值大于0.5就=1，小于0.5就等于0
        if pred[i] >= 0.5:
            pred[i] = int(1)
        else:
            pred[i] = int(0)
    correct = 0
    n = len(pred)
    for i in range(n):
        if pred[i] == target[i]: correct += 1
    score = float(correct / n)
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
    datas = df.loc[:99, :]
    # data, target = datas[:, [0, 1, 2, 3]], datas[:, -1]

    # 划分训练集和测试集
    train, test = Train_test_split(datas)
    # train的索引被打乱了，在预测的过程中用的是np.array数组而不是用的DataFrame，
    # 所以也用不到索引。所以这里重新设置索引，如果不重设索引的话，会导致mini_batch无法从中获得随机索引
    train = train.reset_index(drop=True)
    # 先求权重和偏置,这里的输入是DataFrame格式
    weights, bias = MBGD_LogisticRegression(train)
    print(weights, bias)
    # 划分特征集和标签集
    test_data, test_target = test.iloc[:, [0, 1, 2, 3]], test.iloc[:, -1]
    # 将特征集先做标准化，然后转化为np数组进行预测,这里一定需要做标准化，因为训练集做了标准化，得到的权重和偏置都是标准化之后的值，
    # 如果将原始值输入进去会导致结果发生缩放，造成决策边界失效
    test_data = np.array(Standard(test_data))
    # 将标签转化为一维数组
    test_target = np.array(test_target)
    print('真实值：', test_target)
    score, pred = predict(test_data, test_target, weights, bias)
    print('预测值：', pred)
    print('准确率：', score)
    
    
# 真实值： [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
# 预测值： [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
# 准确率： 1.0
