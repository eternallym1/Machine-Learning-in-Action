import matplotlib.pyplot as plt
from numpy import *
from bi_k_means import *
from support_function import *


def clusterClubs(numClust=5):
    
    # 1 加载地理数据
    dat_list = []
    for line in open("./KMeans/data/places.txt").readlines():
        line_arr = line.split("\t")
        dat_list.append([float(line_arr[4]), float(line_arr[3])])
    dat_mat = mat(dat_list)

    # 2 使用bi-kmeans算法
    my_centroids, my_cluster_assment = biKmeans(dat_mat, numClust, distance_SLC)

    print(my_centroids)

    # 3 绘图
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatter_markers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    img_p = plt.imread('./KMeans/data/Portland.png')
    ax0.imshow(img_p)

    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        data_set_i = dat_mat[nonzero(my_cluster_assment[:,0].A==i)[0],:]
        marker_style = scatter_markers[i]
        ax1.scatter(array(data_set_i[:,0]), array(data_set_i[:,1]), marker=marker_style, s=60)
    ax1.scatter(array(my_centroids[:,0]), array(my_centroids[:,1]), marker='+', s=300)
    
    plt.show()


if __name__ == '__main__':
    clusterClubs()
