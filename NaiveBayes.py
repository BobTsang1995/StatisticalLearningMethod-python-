"""
@File    : NaiveBayes.py
@Time    : 2020-05-17 
@Author  : BobTsang
@Software: PyCharm
@Email   : bobtsang@bupt.edu.cn
"""
# 朴素贝叶斯的思想基础是这样的：对于给出的待分类项，求解在此项出现的条件下各个类别出现的概率，哪个最大，就认为此待分类项属于哪个类别。
### 朴素贝叶斯算法的手工实现（鸢尾花数据集）
# mnist_train:120
# mnist_test:30
# score:0.93
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle as reset

# 我妥协了，还是调了包，划分数据集
def Train_test_split(data, test_size = 0.2, shuffle = True, random_state = None):
    if shuffle:
        data = reset(data, random_state = random_state)

    train = data[int(len(data) * test_size):].reset_index(drop=True)
    test = data[:int(len(data) * test_size)].reset_index(drop=True)

    return train, test

# 用的是高斯朴素贝叶斯，特点是先验为高斯分布（正态分布）
# 先验指的就是假定特征相互独立，每个特征都服从高斯分布
def NaiveBayes_GNB(train,test):
    # 提取训练集的标签种类
    # loc:处理索引中的标签。
    # iloc:处理索引中的位置(因此它只接受整数)。
    labels = train.iloc[:, -1].value_counts().index
    # 存放每个类别的均值
    mean = []
    # 存放每个类别的方差
    std = []
    # 存放测试集的预测结果
    res = []
    # labels：['0', '1', '2']
    for i in labels:
        # 分别提取出每一种类别
        # 注意前行后列        
        item = train.loc[train.iloc[:, -1] == i, :]
        # 当前类别的平均值
        m = item.iloc[:, :-1].mean()
        # 当前类别的方差
        s = np.sum((item.iloc[:, :-1] - m) ** 2) / (item.shape[0])
        # 将当前类别的平均值追加至列表
        mean.append(m)
        # 将当前类别的方差追加至列表
        std.append(s)
    # 变成DataFrame格式，索引为类标签
    means = pd.DataFrame(mean, index=labels)
    stds = pd.DataFrame(std, index=labels)

# 这里应该是由于数据集中的y的种类数量一样多，所以省略了p（y=i），如果不一样多的话，需要求的是p（y=i）x πp（x=xk|y=i）
    for j in range(test.shape[0]):
        # 当前测试实例
        iset = test.iloc[j, :-1].tolist()
         # 正态分布公式
        iprob = np.exp(-((iset - means) ** 2) / (2 * stds ** 2)) / (stds * np.sqrt(2 * np.pi)) 
        # 这里我们传递了一个匿名函数给apply()，当然我们还可以通过axis参数指定轴，比如这里我们可以令axis=1，则就会返回每行的求积
        # 这是个笨办法         
        prob = iprob.apply(lambda x:x['sepal length']*x['sepal width']*x['petal length']*x['petal width'], axis=1)
        # iprob.assign(result=iprob.prod(axis=1))
        # 直接行内求积，加到DataFrame列尾
         # 返回沿轴axis最大值的索引。
        cla = prob.index[np.argmax(prob.values)]  
        # cla = prob.index[np.argmin(prob.values)]
        res.append(cla)
    test['predict'] = res
    return res, test
    # acc = (test.iloc[:, -1] == test.iloc[:, -2]).mean()  # 计算预测准确率
    # print(f'模型预测准确率为{acc}')
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
    print('真实值：',test.iloc[:, -1])
    res, new_test = NaiveBayes_GNB(train, test)
    print('预测值：',res)

    score = Score(new_test)
    print('准确率为：',score)
# 真实值:[1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 1, 2, 0, 1, 2, 1, 1, 1, 1, 1, 0, 2, 2, 0, 1, 0, 0, 0, 2, 2]    
# 预测值:[1, 0, 2, 1, 2, 0, 2, 0, 2, 1, 1, 2, 0, 1, 2, 1, 1, 1, 1, 1, 0, 2, 2, 0, 1, 0, 0, 0, 2, 1]
# 准确率为： 0.9333333333333333
  
