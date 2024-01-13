from numpy import *


def load_data_set(filename):
    """加载数据集

    Args:
        filename (string): 数据集的路径

    Returns:
        numpy.matrix: 数据集（矩阵的形式）
    """
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        current_line = line.strip().split('\t')
        float_line = map(float, current_line) # map: 对序列做映射
        data_mat.append(float_line)
    return mat(data_mat)


def distance_calculation(vec_a, vec_b):
    """距离计算方法: 欧氏距离

    Args:
        vec_a (numpy.array): 向量a
        vec_b (numpy.array): 向量b

    Returns:
        float: 两个向量之间的欧氏距离
    """
    return sqrt(sum(power(vec_a-vec_b, 2)))


def distance_SLC(vec_a, vec_b):
    """
    球面距离计算: 球面余弦定理 - 返回地球表面两点之间的距离
    """
    a = sin(vec_a[0,1]*pi/180) * sin(vec_b[0,1]*pi/180)
    b = cos(vec_a[0,1]*pi/180) * cos(vec_b[0,1]*pi/180) * cos(pi*(vec_b[0,0]-vec_a[0,0])/180)
    return arccos(a+b) * 6371.0


def rand_centroids(data_set, k):
    """随机创建k个簇质心

    Args:
        data_set (numpy.matrix): 数据集
        k (int): 簇质心数量

    Returns:
    numpy.matrix: k个随机的簇质心坐标值
    """
    n = shape(data_set)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(data_set[:, j])
        maxJ = max(data_set[:, j])
        rangeJ = float(maxJ - minJ)
        # 簇质心每一维的坐标值必须在当前维度下最小值和最大值之间
        # random.rand(k, 1)可以保证每个质点同一纬度下的值不同。生成一个k*1的二维数组
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    # print(centroids)
    # print('------------------------------------')
    return centroids


if __name__ == '__main__':
    data_set = mat([[0.1, 0.2, 0.3, 0.4, 0.5],
                    [1.1, 1.2, 1.3, 1.4, 1.5],
                    [2.1, 2.2, 2.3, 2.4, 2.5],
                    [3.1, 3.2, 3.3, 3.4, 3.5],
                    [4.1, 4.2, 4.3, 4.4, 4.5]])
    centroids = rand_centroids(data_set, 3)
    print(centroids)
