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

def distance(pt, arr):
    return norm(pt-arr,axis=1)

def fit_ellipse(p, pts):
    output = least_squares(ellipse.ellipse, p, args=(pts[0,:], pts[1,:]))
    return output.x


if __name__ == '__main__':
    data = io.imread('LaB6_MARCCD.tif')
    edges = feature.canny(data, sigma=0.25)
    X = np.array(np.where(edges)).T
    knn_graph = kneighbors_graph(X, 100, include_self=False)
    n_clusters = 10
    #model = AgglomerativeClustering(linkage=linkage, connectivity=knn_graph, n_clusters=n_clusters)
    #model = SpectralClustering(X)
    model = DBSCAN(eps=10, min_samples=100)
    t0 = time.time()
    model.fit(X)
    elapsed_time = time.time() - t0
    print set(model.labels_)
