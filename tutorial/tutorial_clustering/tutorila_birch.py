from sklearn.cluster import Birch
import matplotlib.pyplot as plt

X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
brc = Birch(n_clusters=None)
brc.fit(X)
label_brc = brc.predict(X)

from sklearn.cluster import SpectralClustering
import numpy as np
X = np.array([[1, 1], [2, 1], [1, 0],
              [4, 7], [3, 5], [3, 6]])
clustering = SpectralClustering(n_clusters=2,
        assign_labels='discretize',
        random_state=0).fit(X)


plt.figure()
plt.hist(label_brc)
plt.figure()
plt.hist(clustering.labels_)
plt.show()