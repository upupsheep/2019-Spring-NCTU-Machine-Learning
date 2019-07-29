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
    """ Calculate Euclidean distance """
    return np.linalg.norm(a - b, axis=ax)


def RBF_kernel(x1, x2, gamma):
    '''
    K(x1, x2) = e^(-gamma*||x1-x2||^2)
    '''
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    K_train = np.zeros((n1, n2))
    x1_norm = np.sum(x1**2, axis=-1)
    x2_norm = np.sum(x2**2, axis=-1)
    if isinstance(x1_norm, np.ndarray) == False:
        x1_norm = np.array([x1_norm])
    if isinstance(x2_norm, np.ndarray) == False:
        x2_norm = np.array([x2_norm])
    dist = x1_norm[:, None] + x2_norm[None, :] - 2 * np.dot(x1, x2.T)
    K_train = np.exp(-gamma * dist)
    return K_train


def kernel_dist(Xj, X, clusters, k, term3, gamma):
    """
    Calculate Kernel Distance
    K(Xj, Xj) - 2 / |Ck| * sum(A_kn*K(Xj, Xn)) + 1 / |Ck|^2 * sum(sum(A_kp*A_kq*K(Xp, Xq)))
    where if the data point Xn is assigned to the k-th cluster, then A_kn = 1
    """
    ### kernel distance from Xj to each centroid ###
    dist = np.zeros((k))

    ### first term ###
    term1 = RBF_kernel(Xj, Xj, gamma)
    ### second term ###
    cluster_cnt = np.array([(clusters == i).sum() for i in range(k)])
    term2 = np.zeros((k))
    for i in range(k):
        term2[i] = 2 / (cluster_cnt[i]) * np.sum(
            RBF_kernel(Xj, X[clusters == i], gamma))
    return term1 - term2 + term3


def init_centroid(X, k, init_type='kmeans-plus'):
    """ return centroid in each clusters (there are k clusters) """
    n = X.shape[0]
    if init_type == 'random':
        idx = np.random.randint(len(X), size=2)
        # idx = np.array([1408, 803])  # circle
        # idx = np.array([123, 1281])  # moon
        C = X[idx]
        return C, idx
    elif init_type == 'kmeans-plus':
        C = []
        C_idx = []
        ### randomly choose one to be the first centroid ###
        idx = np.random.randint(n)
        # idx = np.array([1408, 803])
        # C = np.append(C, X[idx])
        C.append(X[idx])
        C_idx.append(idx)
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
            C_idx.append(next_C_idx)
        C = np.array(C)
        C_idx = np.array(C_idx)
        return C, C_idx


def visualization(X, k, clusters, C):
    global fig_idx
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    global ax
    ax.clear()
    for i in range(k):
        points = X[clusters == i]
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    ax.set_title('Kernel K-Means (Gamma=8)')
    plt.savefig('kernel_kmeans_4k_figures/moon/kernel_kmeans_' + str(fig_idx) +
                '.png')
    fig_idx += 1
    plt.draw()
    plt.pause(0.3)


def kernel_k_means(X, C, C_idx, k):
    """
    Kernel K-means clustering
    Input: X(data points), C(centroid), C_idx(index of centroid), k(# of clusters)
    """
    gamma = 8
    # gamma = 30
    ### To store the value of centroids when it updates ###
    C_old = np.zeros(C.shape)
    ### Use 1d array (clusters) to store cluster of each point ###
    clusters = np.repeat(-1, len(X))
    for i in range(len(C_idx)):
        clusters[C_idx[i]] = i
    ### Distance between new centroids and old centroids ###
    error = dist(C, C_old, None)

    ### Loop untill the error becomes zero ###
    while error != 0:
        ### Precompute kernel distance term 3 ###
        ''' 1 / |Ck|^2 * sum(sum(A_kp*A_kq*K(Xp, Xq))) '''
        term3 = np.zeros((k))
        cluster_cnt = np.array([(clusters == i).sum() for i in range(k)])
        for p in range(k):
            result = 0
            result = np.sum(
                RBF_kernel(X[clusters == p], X[clusters == p], gamma))
            term3[p] = 1 / (cluster_cnt[p]**2) * result

        ### Assigning each value to its closest cluster ###
        new_clusters = np.zeros(clusters.shape)
        for i in range(X.shape[0]):
            distances = kernel_dist(X[i], X, clusters, k, term3, gamma)
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
        print('error: ', error)
        ### Visualization ###
        visualization(X, k, clusters, C)


if __name__ == "__main__":
    data_point = loadData('moon.txt')
    plt.ion()

    # Number of clusters
    k = 4
    # Initialize centroids
    centroid, centroid_idx = init_centroid(data_point, k)
    print(centroid_idx)
    # Calculate kernel K-means
    kernel_k_means(data_point, centroid, centroid_idx, k)
    plt.show()