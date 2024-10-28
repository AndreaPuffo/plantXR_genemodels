import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN, Birch, SpectralClustering

from pydmd import HODMD, DMD, SpDMD

from utils_bechtold import (get_dataset_bechtold, clean_dataset, fit_gaussian_process,
                            plot_scatter_data, get_gp_predictions)
from utils_clustering import getAffinityMatrix, eigenDecomposition

"""
this script computes the DMD for drought and water data, 
then uses the DMD modes matrix to split the dataset into clusters
the DMD is computed as  U A_r V^T, where 
A_r is the reduced order A matrix, 
U and V are the eigenvector matrices
--> use eigenvector matrix as a way to cluster data, since they contain the linear combination of original gene expression 
use HDBSCAN and Spectral Clustering to find possible clusters 
"""

SEED = 167
np.random.seed(SEED)
plot = False


drought_data, water_data = get_dataset_bechtold()

X_train_dr, y_dr, X_finer, y_std_dr = clean_dataset(drought_data)
X_train_wt, y_wt, X_finer, y_std_wt = clean_dataset(water_data)

gp_pred_dr, std_pred_dr = get_gp_predictions(X=X_train_dr, Y=y_dr, Y_std=y_std_dr, X_finer=X_finer)
gp_pred_wt, std_pred_wt = get_gp_predictions(X=X_train_wt, Y=y_wt, Y_std=y_std_wt, X_finer=X_finer)

# update n_genes accordingly
n_genes = gp_pred_dr.shape[0]

# sample from GP with randomicity
noisy_gp_dr = std_pred_dr * 0.75 * np.random.randn(gp_pred_dr.shape[0], gp_pred_dr.shape[1]) + gp_pred_dr
noisy_gp_wt = std_pred_wt * 0.75 * np.random.randn(gp_pred_wt.shape[0], gp_pred_wt.shape[1]) + gp_pred_wt

# snapshot should be of dimension (n_genes x time_steps)
snapshots_dr = noisy_gp_dr
snapshots_wt = noisy_gp_wt
new_time = X_finer[:, 0]

cols_all = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
color_data = cols_all[0]
color_dmd = cols_all[1]
color_spdmd = cols_all[2]
color_hodmd = cols_all[-1]

n_genes_for_plot = 2
genes_for_plot = sorted(np.random.randint(low=0, high=n_genes+1, size=n_genes_for_plot))

#####################
# water and drought data
#####################

rank = 12 # dmd rank (dimension of reduced space)

dmd_wt = DMD(svd_rank=rank)
dmd_wt.fit(snapshots_wt)
ranks_wt = dmd_wt.modes.shape[1]
d_wt = 12

dmd_dr = DMD(svd_rank=rank)
dmd_dr.fit(snapshots_dr)
ranks_dr = dmd_dr.modes.shape[1]
d_dr = 12

# use U matrix (dmd modes) as a basis for clustering
# NOTA: clustering algo does not like imaginary numbers --> use them as extra info by stacking them next to real values
modes_dr = np.hstack([dmd_dr.modes.real, dmd_dr.modes.imag])
modes_wt = np.hstack([dmd_wt.modes.real, dmd_wt.modes.imag])

# cluster
hdb_dr = HDBSCAN()
hdb_dr.fit(modes_dr)
plt.hist(hdb_dr.labels_, bins=len(set(hdb_dr.labels_)))
plt.title(f'Drought data HDBSCAN clusters = {len(set(hdb_dr.labels_))}')

hdb_wt = HDBSCAN()
hdb_wt.fit(modes_wt)
plt.figure()
plt.hist(hdb_wt.labels_, bins=len(set(hdb_wt.labels_)))
plt.title(f'Watered data HDBSCAN clusters = {len(set(hdb_wt.labels_))}')

# cluster w birch
brc = Birch(n_clusters=None)
brc.fit(modes_dr)
labels_brc = brc.predict(modes_dr)

# spectral clustering
affinity_matrix = getAffinityMatrix(modes_dr, k=7)
k, _,  _ = eigenDecomposition(affinity_matrix)

sc = SpectralClustering(n_clusters=k[0],
        assign_labels='discretize',
        random_state=0).fit(modes_dr)

plt.figure()
plt.hist(sc.labels_, bins=len(set(sc.labels_)))
plt.title(f'Drought data spectral clusters = {len(set(sc.labels_))}')

affinity_matrix = getAffinityMatrix(modes_wt, k=7)
k, _,  _ = eigenDecomposition(affinity_matrix)

sc = SpectralClustering(n_clusters=k[0],
        assign_labels='discretize',
        random_state=0).fit(modes_dr)
plt.figure()
plt.hist(sc.labels_, bins=len(set(sc.labels_)))
plt.title(f'Watered data spectral clusters = {len(set(sc.labels_))}')

plt.show()


