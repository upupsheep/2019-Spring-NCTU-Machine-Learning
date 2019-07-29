import matplotlib.pyplot as plt
from matplotlib import style
# style.use('ggplot')
import numpy as np

fig, ax = plt.subplots()
fig_idx = 0


def loadData(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    return data


# Euclidean Distance Caculator
def dist(a, b):
    # print('dist: ', np.linalg.norm(a - b))
    return np.linalg.norm(a - b)


def precomputed_kernel(data, gamma):
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
        # print(x1_norm[:, None])
        dist = x1_norm[:, None] + x2_norm[None, :] - 2 * np.dot(x1, x2.T)
        K_train = np.exp(-gamma * dist)
        return K_train

    return RBF_kernel(data, data, gamma)


class DBSCAN:
    def __init__(self, eps, minPts, data):
        n = data.shape[0]
        self.hasVisit = np.zeros((n))  # (1500,)
        self.clusters = np.repeat(-1, n)  # (1500, )
        self.eps = eps
        self.minPts = minPts
        self.K = precomputed_kernel(data, 9)

    def neighbors(self, point_idx, data):
        """ return all points within P's eps-neighborhood (including P) """
        # neighbors = []
        neighbor_idx = []
        for idx in range(data.shape[0]):
            # if dist(data[idx], point) < self.eps:
            if self.K[idx, point_idx] > self.eps:
                # neighbors.append(data[idx])
                neighbor_idx.append(idx)
        # neighbors = np.array(neighbors)
        neighbor_idx = np.array(neighbor_idx)
        return neighbor_idx

    def expand_cluster(self, point_idx, data, neighbor_idx, C):
        # add point to cluster C
        self.clusters[point_idx] = C
        neighbor_idx_set = set(neighbor_idx)
        # changed = True
        neighbor_idx = neighbor_idx.tolist()
        for i, idx in enumerate(neighbor_idx):
            print(i)

            if self.hasVisit[idx] == 0:  # hasn't been visited
                self.hasVisit[idx] = 1  # mark it as visited
                new_neighbor_idx = self.neighbors(idx, data)
                if new_neighbor_idx.shape[0] >= self.minPts:
                    first_encounter_neighbors = set(
                        new_neighbor_idx).difference(neighbor_idx_set)

                    neighbor_idx.extend(list(first_encounter_neighbors))

                    neighbor_idx_set = neighbor_idx_set.union(
                        set(new_neighbor_idx))
                    #neighbor_idx = np.append(neighbor_idx, new_neighbor_idx)
            if self.clusters[idx] == -1:  # hasn't belong to any cluster
                self.clusters[idx] = C
        return np.array(neighbor_idx)

    def dbscan_main(self, data):
        cluster = -1
        for data_idx in range(data.shape[0]):
            if self.hasVisit[data_idx] == 1:  # has been visited
                continue
            self.hasVisit[data_idx] = 1  # mark data_idx as visited
            neighbor_idx = self.neighbors(data_idx, data)
            if neighbor_idx.shape[0] < self.minPts:
                self.clusters[data_idx] = -2  #  it's noise!!
            else:
                cluster += 1
                neighbor_idx = self.expand_cluster(data_idx, data,
                                                   neighbor_idx, cluster)
            print('clusters:')
            print(self.clusters)
            print(max(self.clusters))
            visualization(data, max(self.clusters) + 1, self.clusters)
        return self.clusters


def visualization(X, k, clusters):
    global fig_idx
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    global ax
    ax.clear()
    plt.xlim((-1.5, 2.3))
    plt.ylim((-2.0, 2.0))
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    # ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    ax.set_title('DBSCAN (eps=0.9, minPts=10)')
    plt.savefig('dbscan_figures/moon/dbscan_' + str(fig_idx) + '.png')
    fig_idx += 1
    plt.draw()
    plt.pause(0.5)
    # plt.show()


if __name__ == "__main__":
    eps = 0.9
    minPts = 10
    data_point = loadData('moon.txt')
    plt.ion()
    """ DBSCAN clustering """
    dbscan = DBSCAN(eps, minPts, data_point)
    clusters = dbscan.dbscan_main(data_point)
    """ visualization """
    k = max(clusters) + 1
    visualization(data_point, k, clusters)
    # plt.show()