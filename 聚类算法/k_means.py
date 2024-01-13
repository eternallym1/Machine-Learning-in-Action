from numpy import *
from support_function import *
import matplotlib.pyplot as plt


def KMeans(data_set, k, distance_calculation=distance_calculation, create_centroids=rand_centroids):
    # 注意：
    # ① 使用随机初始化质心点的方法会导致每次聚类结果不相同
    # ② 如果想返回初始随机质心点，必须使用intial_centroids = centroids.copy()- 创建一个新对象，值与原来对象相同，但是是两个不同的对象！
    # 如果仅仅使用intial_centroids = centroids，那么使用的只是原来对象的引用，那么原来对象怎么变，复制的对象也怎么变
    # 所以一开始没有加copy(), 返回的初始随机质心点与最终质心点始终保持一致！
    m = shape(data_set)[0]
    # 随机生成k个簇质心点
    centroids = create_centroids(data_set, k)
    intial_centroids = centroids.copy()
    centroidAssment = mat(zeros((m, 2)))
    changed_flag = True

    # 当数据集中样本的分配结果发生改变，则循环下面的操作，否则结束循环
    while changed_flag:
        # 计算每一个样本点距离最近的质心点，并记录质心点索引和到质心点的距离的平方
        changed_flag = False
        for i in range(m):
            min_distance = inf
            min_j = inf
            for j in range(k):
                data_i = data_set[i, :]
                centroid_j = centroids[j]
                distance = distance_calculation(data_i, centroid_j)
                if distance < min_distance:
                    min_distance = distance
                    min_j = j
            if float(centroidAssment[i, 0]) != float(min_j):
                # print("yes true")
                # print(centroidAssment[j, 0], min_j)
                changed_flag = True
            centroidAssment[i,:] = [min_j, min_distance]
        # print("--------------------------")
        # print(centroids)

        # 更新质心点
        for j in range(k):
            # 挑选出属于该质心点的所有样本 - 只需要取出所有符合条件的样本点的横向索引值
            data_set_j = data_set[nonzero(centroidAssment[:,0].A==j)[0]]
            # 计算筛选出的样本的每一个维度下的均值作为新的质心点在该维度下的坐标值
            centroids[j,:] = mean(data_set_j, axis=0)

    return centroids, centroidAssment, intial_centroids


if __name__ == '__main__':
    data_set = random.randn(100,2)
    centroids, centroidAssment, intial_centroids = KMeans(data_set, 5)

    plt.figure()
    plt.scatter(array(data_set[:,0]), array(data_set[:,1]), c='y', marker='*')
    plt.scatter(array(centroids[:,0]), array(centroids[:,1]), c='r', marker='+')
    plt.scatter(array(intial_centroids[:,0]), array(intial_centroids[:,1]), c='b')
    plt.show()

"""
① K-means的随机选择初始质心点会导致每次聚类的结果不同, 其只能达到局部最优解
② 如果想复制一个对象, 且生成新的对象, 旧的对象的改变不影响新的对象, 那么需要使用copy()函数。如果没有使用copy对象, 复制的只是引用, 那么原有对象的改变会影响复制的对象的值的改变。
intial_centroids = centroids.copy()
③ random.randn(100,2) - 随机生成一组(100,2),服从标准正态分布的样本
"""
