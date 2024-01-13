from numpy import *
from support_function import *
from k_means import *
import matplotlib.pyplot as plt


# 二分k-均值聚类算法 - 可以达到全局最优解
def biKmeans(dataSet, k, distance_calculation=distance_calculation):
    """
    核心: 一开始将所有数据点划分成一个簇, 然后选择使得总误差最小的簇一分为2, 直至簇的个数等于k
    """
    # 1 初始准备(将所有数据点划分成一个簇，计算整个数据集的质心)
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroids0]

    # 2 计算每一个样本点距离初始质心的距离平方，并将其记录下来
    for j in range(m):
        clusterAssment[j, 1] = distance_calculation(mat(centroids0), dataSet[j, :]) ** 2

    # 3 每一个簇进行k=2的k均值划分，然后挑选出总SSE最小的簇进行簇的重新分配，直至簇质心个数=k
    while len(centList) < k:
        lowestSSE = inf

        # 3.1 对每一个簇进行k=2的k均值划分，并记录总SSE最小的簇信息
        for i in range(len(centList)):
            pstInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A==i)[0], :]
            centroidMat, splitClustAss, _ = KMeans(pstInCurrCluster, 2, distance_calculation)

            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0], 1])
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        # 3.2 将划分后的两个新簇重新分配簇索引（划分簇索引+新加簇索引）
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0], 0] = len(centList)   # 新加簇索引
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0], 0] = bestCentToSplit # 划分簇索引

        # 3.3 将簇划分后的新簇质心添加至簇质心列表中
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1,:])

        # 3.4 将新划分的簇信息（索引+距离平方）同步更新至初始的clusterAssment
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:] = bestClustAss

    myCentroids=list(map(lambda x: [float(x[0]), x[1]], [matrix.tolist(i)[0] for i in centList]))
    centroids_mat=mat(myCentroids)

    return centroids_mat, clusterAssment


if __name__ == '__main__':
    # 每次生成的样本不一致，因此聚类结果也不一致
    # data_set = mat(random.randn(100, 2))
    data_set = mat([[0.1, 0.2],
                [0.1, 0.4],
                [2.1, 2.2],
                [2.1, 2.4],
                [4.1, 4.2],
                [0.1, 0.3],
                [0.1, 0.5],
                [2.1, 2.3],
                [2.1, 2.5],
                [4.1, 4.5]])
    centroids_mat, clusterAssment = biKmeans(data_set, 3)

    plt.figure()
    plt.scatter(array(data_set[:,0]), array(data_set[:,1]), c='r', marker='*')
    plt.scatter(array(mat(centroids_mat)[:,0]), array(mat(centroids_mat)[:,1]), c='g', marker='+')
    plt.show()
