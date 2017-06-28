#! /usr/local/bin/python

import time
import numpy as np
from skimage import io
from skimage import feature
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from numpy.linalg import norm
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN

import sys
import matplotlib.pyplot as plt
import ellipse

if __name__ == '__main__':

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    data = io.imread('LaB6_MARCCD.tif')
    edges = feature.canny(data, sigma=0.25)
    X = np.array(np.where(edges)).T
    knn_graph = kneighbors_graph(X, 150, include_self=False)
    n_clusters = 15
    model = AgglomerativeClustering(linkage='complete', connectivity=knn_graph, n_clusters=n_clusters)
    #model = SpectralClustering(X)
    t0 = time.time()
    model.fit(X)
    elapsed_time = time.time() - t0
    print 'elapsed_time = ' + str(elapsed_time)

    ypred = model.labels_.astype(np.int)
    print (ypred)
    plt.imshow(data)
    plt.scatter(X[:,0], X[:,1], color=colors[ypred].tolist(), s=10)
    plt.show()
