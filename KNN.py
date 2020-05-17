"""
@File    : KNN.py
@Time    : 2020-05-15 
@Author  : BobTsang
@Software: PyCharm
@Email   : bobtsang@bupt.edu.cn
"""
### K近邻算法的手工实现（鸢尾花数据集）
# mnist_train:105
# mnist_test:45
# score:0.91

### 代码
import pandas as pd
import numpy as np
from sklearn import datasets

# 手写实现sklearn打乱数据集函数
def Random_number(data_size):
    """
        该函数使用shuffle()打乱一个包含从0到数据集大小的整数列表。因此每次运行程序划分不同，导致结果不同

        改进：
        可使用random设置随机种子，随机一个包含从0到数据集大小的整数列表，保证每次的划分结果相同。

        :param data_size: 数据集大小
        :return: 返回一个列表
    """
    num_set = []
    for i in range(data_size):
        num_set.append(i)
    random.shuffle(num_set)

    return num_set

# 手写实现sklearn划分数据集
def Train_test_split(data_set, target_set, size=0.3):
    """
        分割数据集，我这里默认数据集的0.3是测试集

        :param data_set: 数据集
        :param target_data: 标签数据
        :param size: 测试集所占的大小
        :return: 返回训练集数据、训练集标签、训练集数据、训练集标签
    """
    # 计算训练集的数据个数
    train_size = int((1-size) * len(data_set))

    # 获得数据
    data_index = Random_number(len(data_set))

    # 分割数据集（X表示数据，y表示标签），以返回的index为下标
    x_train = data_set[data_index[:train_size]]

    x_test = data_set[data_index[train_size:]]

    y_train = target_set[data_index[:train_size]]

    y_test = target_set[data_index[train_size:]]
    return x_train, x_test, y_train, y_test   
    
def Distance(x_test, x_train):
    """
        :param x_test: 测试集
        :param x_train: 训练集
        :return: 返回计算的距离
    """
    # sqrt_x = np.linalg.norm(test-train)  # 使用norm求二范数（距离）
    # 这里用欧式距离
    distances = np.sqrt(sum(x_test - x_train) ** 2)
    return distances
    
def KNN(x_test, x_train, y_train, k):
    """
       :param x_test: 测试集数据
       :param x_train: 训练集数据
       :param y_train: 测试集标签
       :param k: 邻居数
       :return: 返回一个列表包含预测结果
    """
    # 预测结果列表，用于存储测试集预测出来的结果
    pred_res = []

    # 训练集的长度
    train_size = len(x_train)

    # 创建一个全零的矩阵，长度为训练集的长度
    distances = np.array(np.zeros(train_size))

    # 计算每一个测试集与每一个训练集的距离
    for i in x_test:
        for index in range(train_size):

            # 计算数据之间的距离
            distances[index] = Distance(i, x_train[index])

        # 排序后的距离的下标
        # argsort函数返回的是从小到大的距离
        sorted_dis = np.argsort(distances)

        # 创建一个哈希表用于储存0，1，2出现的次数，用于计算准确率
        class_count = {}

        # 取出k个最短距离
        for i in range(k):

            # 获得下标所对应的标签值
            sort_label = y_train[sorted_dis[i]]

            # 将标签存入哈希表之中并存入个数
            class_count[sort_label] = class_count.get(sort_label, 0) + 1
        # 对标签进行排序
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        # 将出现频次最高的放入预测结果列表
        pred_res.append(sorted_class_count[0][0])
    # 返回预测结果列表
    return pred_res

def Score(pred_res, y_test):
    """
       :param pred_res: 预测结果列表
       :param y_test: 测试集标签
       :return: 返回测试集精度
    """
    cnt = 0
    for i in range(len(pred_res)):
        if pred_res[i] == y_test[i]:
            cnt += 1

    score = cnt / len(pred_res)
    return score


if __name__ == "__main__":
    ### 对数据进行预处理
    # 从sklearn中读取到数据集
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = Train_test_split(iris.data, iris.target)
    
    # # 用sklearn库实现数据归一化和标准化，这里想用一下标准化和归一化，结果发现得分变低，所以弃用了
    # ss = StandardScaler()
    # x_train = ss.fit_transform(x_train)
    # x_test = ss.fit_transform(x_test)
    # mms = MinMaxScaler()
    # x_train = mms.fit_transform(x_train)
    # x_test = mms.fit_transform(x_test)
    res = KNN(x_test, x_train, y_train, 6)
    print('真实值：', y_test)
    print('预测值：', np.array(res))
    score = Score(res, y_test)
    print('测试集的精度：%.2f' % score)
    
# 真实值： [0 1 2 2 0 1 2 2 1 0 2 0 2 1 0 0 0 1 1 1 1 1 1 2 0 2 1 2 1 0 0 2 1 2 2 1 2
#  1 0 0 1 1 1 2 1]
# 预测值： [0 1 2 2 0 1 2 2 1 0 2 0 2 1 0 0 0 1 1 1 1 1 2 2 0 2 1 1 1 0 0 2 1 1 2 1 2
#  2 0 0 1 1 1 2 1]
# 测试集的精度：0.91
