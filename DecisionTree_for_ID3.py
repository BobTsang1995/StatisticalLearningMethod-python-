"""
@File    : DecisionTree.py
@Time    : 2020-05-18 
@Author  : BobTsang
@Software: PyCharm
@Email   : bobtsang@bupt.edu.cn
"""
# 该决策树代码只涉及ID3算法，C4.5和CART没时间解释了，赶紧上车
# 	ID3：使用信息增益进行特征选择
# 	C4.5：信息增益率
# 	CART：基尼系数 
# 	一个特征的信息增益(或信息增益率，或基尼系数)越大，表明特征对样本的熵的减少能力更强（越快到达叶子结点），这个特征使得数据由不
# 确定性到确定性的能力越强。所以选分裂点要选该节点衡量系数最大的。

import numpy as np
import pandas as pd
from sklearn import datasets

# 划分数据集,将数据集划分为训练集和测试集
# # 写法一：普通方法划分数据集
def Train_test_split(data, size=0.2):

    # 打乱索引
    index = [i for i in range(len(data))]
    # 这里的随机种子我是试出来的，因为训练集和测试集分布不均匀，所以我也没有什么特别好的办法。     
    np.random.seed(12)
    np.random.shuffle(index)

    # 以8：2的比例划分为训练集和测试集
    train = data.loc[index[:int(len(data) * (1-size))]]
    test = data.loc[index[int(len(data) * (1-size)):]]

    return train, test
# 这个方法感觉有问题，因为加入了新的特征列，导致决策树无法构造下去。。。所以还是不用了
# # #写法二：写法一可能会导致数据集划分不均匀，主要是训练集和测试集的类别Y的类别会不均匀
# # 类似于分层抽样，每个类别随机划分同样个数的样本给训练集
# def Step_sampling(data):
#     train_list = []
#     test_list = []
#
#     for i in range(data.iloc[:, -1].value_counts().shape[0]):
#         subdata = data.loc[i * 50:(i + 1) * 50 -1]
#         # 重置索引
#         subdata = subdata.reset_index()
#         subtrain, subtest = Train_test_split(subdata)
#         train_list.append(subtrain)
#         test_list.append(subtest)
#
#     # 合并 DataFrame，ignore_index=True 表示忽略原索引，类似重置索引的效果
#     train = pd.concat(train_list, ignore_index=True)
#     test = pd.concat(test_list, ignore_index=True)
#
#     return train, test

# 计算经验熵
def caculateEntropy(data):
    N = data.shape[0]
    # 将Y的取值全部提取出来
    category = data.iloc[:, -1].value_counts().values
    p = category / N
    # 经验熵公式
    entropy = (-p * np.log2(p)).sum()
    return entropy

# 计算离散信息增益，当属性为离散值时，即统计学习方法书中A1属性为青年，中年，老年
def caculateInfoGain(data, attribute):
    # 计算经验熵
    baseEntropy = caculateEntropy(data)
    # 提取出当前特征列的所有取值，如：青年，中年，老年
    levels = data.iloc[:, attribute].value_counts().index
    # 初始化子节点的条件熵
    sumentropy = 0
    # 对当前列的每一个取值进行循环
    for D in levels:
        # 当前特征的某一个子集的dataframe
        child = data[data.iloc[:, attribute] == D]
        # 计算某一个子节点的信息熵
        entropy = caculateEntropy(child)
        # 计算当前列的经验条件熵，得到单特征情况下的Y的熵和pi的乘积，即经验条件熵p[D|Ai]
        sumentropy += (child.shape[0] / data.shape[0]) * entropy
        # 计算当前特征列的信息增益
    infoGain = baseEntropy - sumentropy
    return infoGain

# 特征列的最优选择
def Bestselect(data):
    # 初始化信息增益
    bestGain = 0
    # 初始化最佳切分列，标签列
    axis = -1
    # 循环得到离散特征attribute划分的数据集的信息增益
    for attribute in range(data.shape[1]-1):
        infoGain = caculateInfoGain(data, attribute)
        if infoGain > bestGain:
            # 更新最优选择
            bestGain = infoGain
            # 更新最优选的特征所在列的索引，作为子节点
            axis = attribute
    return bestGain, axis

# 删除已经选择的最优子节点（特征）所在的列,得到子数据集，用于查找下一个最优选择
# 来构建下一个子树
def splitdata(data, axis, value):
    col = data.columns[axis]
    subdata = data.loc[data[col] == value, :].drop(col, axis=1)
    return subdata

# 构造决策树
def CreateDcTree(data):
    # 获得数据集的所有特征得到特征集列表
    features = list(data.columns)
    # 得到所有的分类和统计量，其中index为所有类别的值，values对应统计量
    category = data.iloc[:, -1].value_counts()
    '''
    # 如果特征集为空，则该树为单节点树
    # 计算数据集中出现次数最多的标签
    if not features:
        # 返回沿轴axis最大值的索引。
        return category.index[np.argmax(category.values, axis=0)]
    '''

    # 判断最多标签数目是否等于数据集行数，即数据集中，只包同一种标签
    # 或者数据集是否只有一列，即特征集为空，该树为单节点树
    # 如果为True，则该数据集熵为0，没有信息增益，则该树为单节点树
    if category.iat[0] == data.shape[0] or data.shape[1] == 1:
        # 返回类标签
        return category.index[0]
    # 定义阈值ϵ
    threshold = 0
    # 计算信息增益，如果最大信息增益小于阈值，则该树为单节点树
    bestGain, axis = Bestselect(data)
    if threshold > bestGain:
        return category.index[0]

    # 选取特征子集
    bestfeature = features[axis]
    #
    # 采用字典嵌套的方式存储树信息，循环最优选择的特征列中的所有特征，储存信息
    # 构造一棵用字典来表示的树，字典中存储的信息为子树的信息
    DecisionTree = {bestfeature: {}}
    # 删除特征集列表中已被选为最优的特征，对剩下的特征继续进行最优选择
    del features[axis]
    # 提取最优选择特征列所有属性值
    feature_value = set(data.iloc[:, axis])
    # 递归构造一个决策树
    for value in feature_value:
        DecisionTree[bestfeature][value] = CreateDcTree(splitdata(data, axis, value))
    return DecisionTree

# 输入测试集特征和决策树，根据决策树树来判断分类
def classify(DecisionTree, labels, testVec):
    # 得到决策树的根节点
    # 迭代得到所有节点信息
    root = next(iter(DecisionTree))

    # 根据根节点找到子节点的字典
    childDict = DecisionTree[root]

    # 根节点对应特征列的列索引，即特征索引
    featureIndex = labels.index(root)

    for key in childDict.keys():
        if testVec[featureIndex] == key:
            if type(childDict[key]) == dict:
                return classify(childDict[key], labels, testVec)

            else:
                return childDict[testVec[featureIndex]]

# 预测分类
def predict(train, test):
    # 根据训练集生成一棵决策树
    DecisionTree = CreateDcTree(train)
    print(DecisionTree)
    # 找出训练集集所有的列名称
    labels = list(train.columns)
    res = []
    # 遍历测试集的所有行
    for i in range(test.shape[0]):
        # 测试集中的一个实例
        testVec = test.iloc[i, :-1]
        classlabel = classify(DecisionTree, labels, testVec)
        res.append(classlabel)
    # 将预测结果追加到测试集最后一列
    test['predict'] = res
    return res, test
# 计算准确率
def Score(test):
    cnt = 0
    true_label = np.array(test.iloc[:, -2])
    pred_label = np.array(test.iloc[:, -1])
    for i in range(len(true_label)):
        if true_label[i] == pred_label[i]:
            cnt += 1
    return float(cnt / len(true_label))

if __name__ == '__main__':
    iris = datasets.load_iris()
    # 创建一个dataframe表型数据结构
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    # 列尾添加新的一列'label', 值为iris.target(Series对象)
    data['label'] = iris.target

    # 重命名列名
    data.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    train, test = Train_test_split(data)
    print(train)
    print(test)

    print('真实值：', np.array(test.iloc[:, -1]))
    res, test_pred = predict(train, test)

    print('预测值：', res)
    score = Score(test_pred)
    print('准确率：', score)
# 真实值： [0, 2, 0, 2, 0, 0, 1, 2, 2, 1, 2, 2, 0, 1, 1, 0, 2, 2, 2, 1, 2, 2, 2, 0, 0, 1, 0, 2, 2, 1]
# 预测值： [0, 2, 0, 2, 0, 0, 1, 2, 2, 1, 1, 2, 0, 1, 1, 0, 2, 2, 2, 1, 2, 2, 1, 0, 0, 1, 0, 2, 1, 1]
# 准确率： 0.9
