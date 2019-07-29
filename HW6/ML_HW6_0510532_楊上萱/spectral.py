import matplotlib.pyplot as plt
from matplotlib import style
# style.use('ggplot')
import numpy as np

fig, ax = plt.subplots()
fig_idx = 0


def loadData(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    return data


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


def visualization(X, k, clusters, C):
    global fig_idx
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    global ax
    ax.clear()
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    # ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    ax.set_title('Spectral Clustering (Gamma=30)')
    # plt.savefig('spectral_eigenvector/moon/spectral_eig_' + str(fig_idx) +
    # '.png')
    fig_idx += 1
    plt.draw()
    plt.pause(0.5)
    # plt.show()


def dist(a, b, ax=1):
    """ Calculate Euclidean distance """
    return np.linalg.norm(a - b, axis=ax)


def init_centroid(X, k):
    """ return centroid in each clusters (there are k clusters) """
    C = X[np.random.choice(X.shape[0], k, replace=False), :]
    return C


def k_means(data_point, X, C, k):
    ### To store the value of centroids when it updates ###
    C_old = np.zeros(C.shape)
    ### Use 1d array (clusters) to store cluster of each point ###
    clusters = np.zeros(len(X))
    ### Distance between new centroids and old centroids ###
    error = dist(C, C_old, None)

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
        print('error: ', error)
        ### Visualization ###
        visualization(data_point, k, clusters, C)
    # visualization(data_point, k, clusters, C)


def similarity_graph(data):
    gamma = 30
    n = data.shape[0]  # 1500
    G = RBF_kernel(data, data, gamma)
    return G


def laplacian(W):
    n = G.shape[0]  # 1500

    def degree_matrix():
        # D = np.zeros((n, n))
        # for i in range(n):
        #     for j in range(n):
        #         if (i != j):
        #             D[i][j] += W[i][j]
        return np.diag(np.sum(W, axis=1))

    return degree_matrix() - W


def find_k_smallest_eigenvalues(L, K):
    eigen_value, eigen_vector = np.linalg.eig(L)
    sorting_index = np.argsort(eigen_value)
    eigen_value = eigen_value[sorting_index]
    eigen_vector = eigen_vector.T[sorting_index]
    return eigen_value[1:k + 1], (eigen_vector[0:k])


if __name__ == "__main__":
    k = 2
    data_point = loadData('circle.txt')
    plt.ion()
    """ Construct a similarity graph G """
    G = similarity_graph(data_point)
    # print(G[2][3])
    # print(G[3][2])
    print(G)
    """ Compute the unnormalized Laplacian L """
    L = laplacian(G)
    print(L)
    """ Compute the first k eigenvectors u1, . . . ,uk of L. """
    eigen_value, eigen_vector = find_k_smallest_eigenvalues(L, k)
    print('eigen_value:')
    print(eigen_value)
    print('eigen_vector:')
    print(eigen_vector[0])
    #print(L @ eigen_vector.T)
    """ Cluster the points (yi)i=1,...,n in Rk with the k-means algorithm into clusters C1, . . . , Ck. """
    C = init_centroid(eigen_vector.T, k)
    print(C)
    k_means(data_point, eigen_vector.T, C, k)
    plt.show()
