"""
@File    : AdaBoost.py
@Time    : 2020-05-26 
@Author  : BobTsang
@Software: PyCharm
@Email   : bobtsang@bupt.edu.cn
"""
# 这次用的是乳腺癌数据集做的二分类任务，因为鸢尾花数据集太小，特征较少，对于提升树不太cover
# Minst:596x31
# time:62s

import pandas as pd
import numpy as np
from sklearn import datasets
import random
import time


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
    random.seed(1)
    for i in range(data_size):
        num_set.append(i)
    random.shuffle(num_set)
    return num_set


def Train_test_split(data_set, target_data, size=0.2):
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
    
def Caculation_error_Gx(x_train, y_train, n, div, rule, D):
    """
     计算分类错误率
     :param x_train:训练集数据
     :param y_trian:训练集标签
     :param n:要操作的特征
     :param div:划分点（阈值）
     :param rule:正反例标签
     :param D:权值分布
     :return:预测结果，分类误差
    """
    # 初始化分类误差率为0
    error = 0
    # 将训练数据矩阵中特征为n的那一列单独剥出来做成数组。因为其他元素我们并不需要，
    # 直接对庞大的训练集进行操作的话会很慢
    x = x_train[:, n]
    # 同样将标签也转换成数组格式，x和y的转换只是单纯为了提高运行速度
    # 测试过相对直接操作而言性能提升很大
    y = y_train
    predict = []

    # 依据小于和大于的标签依据实际情况会不同，在这里直接进行设置
    if rule == 'LisOne':    L = 1; H = -1
    else:                   L = -1; H = 1

    # 遍历所有样本的特征m
    for i in range(x_train.shape[0]):
        if x[i] < div:
            # 如果小于划分点，则预测为L
            # 如果设置小于div为1，那么L就是1，
            # 如果设置小于div为-1，L就是-1
            predict.append(L)
            # 如果预测错误，分类错误率要加上该分错的样本的权值（8.1式）
            if y[i] != L:
                error += D[i]
        elif x[i] >= div:
            # 与上面思想一样
            predict.append(H)
            if y[i] != H:
                error += D[i]
    # 返回预测结果和分类错误率e
    # 预测结果其实是为了后面做准备的，在算法8.1第四步式8.4中exp内部有个Gx，要用在那个地方
    # 以此来更新新的D
    return np.array(predict), error

def CreateSingleBoostingTree(x_train, y_train, D):
    """
    创建单层提升树
    :param x_train:训练数据集
    :param y_train:训练标签集
    :param D:权值分布
    :return:单层提升树
    """

    # 获得样本数目及特征数量
    m, n = np.shape(x_train)
    # 单层树的字典，用于存放当前层提升树的参数
    # 也可以认为该字典代表了一层提升树
    singleBoostTree = {}
    # 初始化分类误差率，分类误差率在算法8.1步骤（2）（b）有提到
    # 误差率最高也只能100%，因此初始化为1
    singleBoostTree['error'] = 1

    # 对每一个特征进行遍历，寻找用于划分的最合适的特征
    for i in range(n):
        # 因为特征已经经过二值化，只能为0和1，因此分切分时分为-0.5，0.5，1.5三种进行切割
        for div in [-0.5, 0.5, 1.5]:
            # 在单个特征内对正反例进行划分时，有两种情况：
            # 可能是小于某值的为1，大于某值得为-1，也可能小于某值得是-1，反之为1
            # 因此在寻找最佳提升树的同时对于两种情况也需要遍历运行
            # LisOne：Low is one：小于某值得是1
            # HisOne：High is one：大于某值得是1
            for rule in ['LisOne', 'HisOne']:
                # 按照第i个特征，以值div进行切割，进行当前设置得到的预测和分类错误率
                Gx, error = Caculation_error_Gx(x_train, y_train, i, div, rule, D)
                # 如果分类错误率e小于当前最小的e，那么将它作为最小的分类错误率保存
                if error < singleBoostTree['error']:
                    singleBoostTree['error'] = error
                    # 同时也需要存储最优划分点、划分规则、预测结果、特征索引
                    # 以便进行D更新和后续预测使用
                    singleBoostTree['div'] = div
                    singleBoostTree['rule'] = rule
                    singleBoostTree['Gx'] = Gx
                    singleBoostTree['feature'] = i
    # 返回单层的提升树
    return singleBoostTree

def CreateBoostingTree(x_train, y_train, treeNum = 50):
    """
    创建提升树
    创建算法依据“8.1.2 AdaBoost算法” 算法8.1
    :param x_train:训练数据集
    :param y_train:训练标签
    :param treeNum:树的层数
    :return:提升树
    """
    # 将数据和标签转化为数组形式
    trainDataArr = np.array(x_train)
    trainLabelArr = np.array(y_train)
    # 没增加一层数后，当前最终预测结果列表
    finalpredict = [0] * len(trainLabelArr)
    # 获得训练集数量以及特征个数
    m, n = np.shape(trainDataArr)

    # 依据算法8.1步骤（1）初始化D为1/N
    D = [1 / m] * m
    # 初始化提升树列表，每个位置为一层
    tree = []
    # 循环创建提升树
    for i in range(treeNum):
        # 得到当前层的提升树
        curTree = CreateSingleBoostingTree(trainDataArr, trainLabelArr, D)
        # 根据式8.2计算当前层的alpha
        alpha = 1 / 2 * np.log((1 - curTree['error']) / curTree['error'])
        # 获得当前层的预测结果，用于下一步更新D
        Gx = curTree['Gx']
        # 依据式8.4更新D
        # 考虑到该式每次只更新D中的一个w，要循环进行更新知道所有w更新结束会很复杂（其实
        # 不是时间上的复杂，只是让人感觉每次单独更新一个很累），所以该式以向量相乘的形式，
        # 一个式子将所有w全部更新完。
        # 该式需要线性代数基础，如果不太熟练建议补充相关知识，当然了，单独更新w也一点问题没有
        # np.multiply(trainLabelArr, Gx)：exp中的y*Gm(x)，结果是一个行向量，内部为yi*Gm(xi)
        # np.exp(-1 * alpha * np.multiply(trainLabelArr, Gx))：上面求出来的行向量内部全体
        # 成员再乘以-αm，然后取对数，和书上式子一样，只不过书上式子内是一个数，这里是一个向量
        # D是一个行向量，取代了式中的wmi，然后D求和为Zm
        # 书中的式子最后得出来一个数w，所有数w组合形成新的D
        # 这里是直接得到一个向量，向量内元素是所有的w
        # 本质上结果是相同的
        D = np.multiply(D, np.exp(-1 * alpha * np.multiply(trainLabelArr, Gx))) / sum(D)
        # 在当前层参数中增加alpha参数，预测的时候需要用到
        curTree['alpha'] = alpha
        # 将当前层添加到提升树索引中。
        tree.append(curTree)

        # -----以下代码用来辅助，可以去掉---------------
        # 根据8.6式将结果加上当前层乘以α，得到目前的最终输出预测
        finalpredict += alpha * Gx
        # 计算当前最终预测输出与实际标签之间的误差
        error = sum([1 for i in range(len(x_train)) if np.sign(finalpredict[i]) != trainLabelArr[i]])
        # 计算当前最终误差率
        finalError = error / len(x_train)
        # 如果误差为0，提前退出即可，因为没有必要再计算算了
        if finalError == 0:
            return tree
        # 打印一些信息
        print('iter:%d:%d, single error:%.4f, final error:%.4f'%(i, treeNum, curTree['error'], finalError))
    # 返回整个提升树
    return tree

def predict(x, div, rule, feature):
    """
    输出单层的预测结果
    :param x:预测样本
    :param div:划分点
    :param rule:划分规则
    :param feature:进行操作的特征
    :return:
    """
    #依据划分规则定义小于及大于划分点的标签
    if rule == 'LisOne':
        L = 1; H = -1
    else:
        L = -1; H = 1

    #判断预测结果
    if x[feature] < div:
        return L
    else:
        return H

def model_test(x_test, y_test, tree):
    """
    测试模型
    :param x_test:测试数据集
    :param y_test:测试标签集
    :param tree:提升树
    :return:准确率
    """
    # 错误率计数值
    errorCnt = 0
    # 遍历每一个测试样本
    for i in range(len(x_test)):
        # 预测结果值，初始为0
        res = 0
        # 依据算法8.1式8.6
        # 预测式子是一个求和式，对于每一层的结果都要进行一次累加
        # 遍历每层的树
        for curTree in tree:
            # 获取该层参数
            div = curTree['div']
            rule = curTree['rule']
            feature = curTree['feature']
            alpha = curTree['alpha']
            # 将当前层结果加入预测中
            res += alpha * predict(x_test[i], div, rule, feature)
        #预测结果取sign值，如果大于0 sign为1，反之为0
        if np.sign(res) != y_test[i]:
            errorCnt += 1
    #返回准确率
    return float(1 - errorCnt / len(x_test))

# 将所有数据（不包括标签）进行二值化处理
def find_init_div(data):
    inMat = data.copy()
    # 求每一列的均值
    # axis=0意味着取这列的每一行出来求均值，最终得到所有的列的均值
    inMeans = np.mean(inMat, axis=0)
    # 每一个特征属性标准化
    inMat = inMat - inMeans
    inMat = inMat.applymap(lambda x: int(0) if x <= 0 else 1)
    inMat = np.array(inMat)
    return inMat

if __name__ == '__main__':
    # 开始时间
    start = time.time()

    # 获取训练集和测试集
    breastcancer = datasets.load_breast_cancer()
    # print(breastcancer)
    # 创建一个dataframe表型数据结构
    df = pd.DataFrame(breastcancer.data, columns=breastcancer.feature_names)
    # 列尾添加新的一列'label', 值为iris.target(Series对象)
    df['label'] = breastcancer.target
    print(df)

    # 找到训练数据集的初始阈值，依据初始阈值将数据进行二值化处理，大于v的转换成1，小于v的转换成0，方便后续计算
    # 打乱数据

    data = find_init_div(df.iloc[:, :-1])
    print(data)

    target = np.array(df.iloc[:, -1])
    print(target)
    x_train, x_test, y_train, y_test = Train_test_split(data, target)
    # print(x_train)
    # print(x_test)

    # 转换为二分类任务
    # 替换标签把0，1换成-1，1
    y_train = np.array([int(1) if i == 1 else int(-1) for i in y_train])
    # print(x, y)
    # 替换标签把0，1换成-1，1
    y_test = np.array([int(1) if i == 1 else int(-1) for i in y_test])

    # 创建提升树
    print('start init train')
    tree = CreateBoostingTree(x_train, y_train, 100)

    # 测试
    print('start to test')
    accuracy = model_test(x_test, y_test, tree)
    print('the accuracy is:%.4f' % (accuracy * 100), '%')
    print(accuracy)

    # 结束时间
    end = time.time()
    print('time span:', end - start)
    
    # the accuracy is:97.0760 %
