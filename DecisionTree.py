"""
@File    : DecisionTree.py
@Time    : 2020-05-18 
@Author  : BobTsang
@Software: PyCharm
@Email   : bobtsang@bupt.edu.cn
"""
# 划分数据集,将数据集划分为训练集和测试集
# # 写法一：普通方法划分数据集
def Train_test_split(data, size=0.2):

    # 打乱索引
    index = [i for i in range(len(data))]
    np.random.seed(1)
    np.random.shuffle(index)

    # 以8：2的比例划分为训练集和测试集
    train = data.loc[index[:int(len(data) * (1-size))]]
    test = data.loc[index[int(len(data) * (1-size)):]]

    return train, test

# #写法二：写法一可能会导致数据集划分不均匀，主要是训练集和测试集的类别Y的类别会不均匀
# 类似于分层抽样，每个类别随机划分同样个数的样本给训练集
def Step_sampling(data):
    train_list = []
    test_list = []

    for i in range(data.iloc[:, -1].value_counts().shape[0]):
        subdata = data.loc[i * 50:(i + 1) * 50 -1]
        # 重置索引
        subdata = subdata.reset_index()
        subtrain, subtest = Train_test_split(subdata)
        train_list.append(subtrain)
        test_list.append(subtest)

    # 合并 DataFrame，ignore_index=True 表示忽略原索引，类似重置索引的效果
    train = pd.concat(train_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)

    return train, test

# 计算经验熵
def calEntropy(data):
    N = data.shape[0]
    # 将Y的取值全部提取出来
    category = data.iloc[:, -1].value_counts().values
    p = category / N
    # 经验熵公式
    entropy = (-p * np.log2(p)).sum()
    return entropy
    
# 计算离散信息增益，当属性为离散值时，即统计学习方法书中A1属性为青年，中年，老年
def BestInformationDiscreteGain(data):
    # 计算经验熵
    baseEntropy = calEntropy(data)
    # 得到离散特征attribute划分的数据集的信息增益
    # 得到当前特征列的所有label的取值
    # 初始化信息增益
    bestGain = 0
    axis = -1  # 初始化最佳切分列，标签列
    for attribute in range(data.shape[1]-1):
        # 提取出当前特征列的所有取值，如：青年，中年，老年
        levels = data.iloc[:, attribute].value_counts().index
        # 初始化子节点的条件熵
        sumentropy = 0
        # 对当前列的每一个取值进行循环
        for D in levels:
            # 当前特征的某一个子集的dataframe
            child = data[data.iloc[:, attribute] == D]
            # 计算某一个子节点的信息熵
            entropy = calEntropy(child)
            # 计算当前列的经验条件熵，得到单特征情况下的Y的熵和pi的乘积，即经验条件熵p[D|Ai]
            sumentropy += (child.shape[0] / data.shape[0]) * entropy

        # 计算当前特征列的信息增益
        infoGain = baseEntropy - sumentropy

        if infoGain > bestGain:
            # 更新最优选择
            bestGain = infoGain
            # 更新最优选的特征所在列的索引，作为子节点
            axis = attribute

    return axis
    
# 删除已经选择的最优子节点（特征）所在的列
def delete_leavenode(data, axis, value):
    col = data.columns[axis]
    redata = data.loc[data[col] == value, :].drop(col, axis=1)
    return redata
    
# 创建树
def CreateDecisionTree(data):
    # 获得数据集的所有特征
    features = list(data.columns)
    # 得到所有的分类和统计量，其中index为所有类别的值，values对应统计量
    category = data.iloc[:, -1].value_counts()
    # 判断最多标签数目是否等于数据集行数，或者数据集是否只有一列
    # 如果为True，则该数据集熵为0，没有信息增益，不能成一棵树
    if category.iat[0] == data.shape[0] or data.shape[1] == 1:
        # 返回类标签
        return category.index[0]
    # 确定出当前最优选择特征列的索引
    axis = BestInformationDiscreteGain(data)
    # 获取该索引对应的特征
    bestfeat = features[axis]
    # 采用字典嵌套的方式存储树信息
    DecisionTree = {bestfeat: {}}
    del features[axis]
    # 提取最优选择特征列所有属性值
    valueset = set(data.iloc[:, axis])
    # 对每一个属性值递归建树
    for value in valueset:
        DecisionTree[bestfeat][value] = CreateDecisionTree(delete_leavenode(data, axis, value))
    return DecisionTree
