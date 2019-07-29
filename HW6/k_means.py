import matplotlib.pyplot as plt
from matplotlib import style
# style.use('ggplot')
import numpy as np

fig, ax = plt.subplots()
fig_idx = 0


def loadData(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    return data


def dist(a, b, ax=1):
    """ calculate euclidean distance """
    return np.linalg.norm(a - b, axis=ax)


def init_centroid(X, k, init_type='random'):
    """ return centroid in each clusters (there are k clusters) """
    n = X.shape[0]
    if init_type == 'random':
        C = X[np.random.choice(X.shape[0], k, replace=False), :]
        return C
    elif init_type == 'kmeans-plus':
        C = []
        ### randomly choose one to be the first centroid ###
        idx = np.random.randint(n)
        # C = np.append(C, X[idx])
        C.append(X[idx])
        ### Use shortest distance to be the weight and find next centroid ###
        while len(C) < k:
            D = []
            for i in range(n):
                d_to_centroids = np.array(
                    [dist(X[i], C[j], None) for j in range(len(C))])
                D.append(np.min(d_to_centroids))
            D = np.array(D)
            D_weight = D / np.sum(D)
            next_C_idx = np.random.choice(n, p=D_weight)
            C.append(X[next_C_idx])
        C = np.array(C)
        return C


def visualization(X, k, clusters, C):
    global fig_idx
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    global ax
    ax.clear()
    for i in range(k):
        points = X[clusters == i]
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    ax.set_title('K-Means++')
    plt.savefig('k_means_plus_4k_figures/circle/kmeans_plus_4k_' +
                str(fig_idx) + '.png')
    fig_idx += 1
    plt.draw()
    plt.pause(0.3)


def k_means(X, C, k):
    """
    K-means clustering
    Input: X(data points), C(centroid), k(# of clusters)
    """

    ### To store the value of centroids when it updates ###
    C_old = np.zeros(C.shape)
    ### Use 1d array (clusters) to store cluster of each point ###
    clusters = np.zeros(len(X))
    ### Distance between new centroids and old centroids ###
    error = dist(C, C_old, None)
    # error = np.linalg.norm(C - C_old)
    # print(error.shape)

    ### Loop untill the error becomes zero ###
    while error != 0:
        ### Assigning each value to its closest cluster ###
        new_clusters = np.zeros(clusters.shape)
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            new_clusters[i] = cluster
        clusters = new_clusters
        ### Store the old centroid values ###
        C_old = C.copy()
        #### Find new centroids by mean of each cluster ###
        for i in range(k):
            points = X[clusters == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)
        print(error)
        ### Visualization ###
        visualization(X, k, clusters, C)


if __name__ == "__main__":
    data_point = loadData('circle.txt')
    plt.ion()
    # Number of clusters
    k = 4
    # Initialize centroid
    # centroid = init_centroid(data_point, k)
    centroid = init_centroid(data_point, k, 'kmeans-plus')
    print(centroid)
    # Calculate K-means
    k_means(data_point, centroid, k)
    plt.show()