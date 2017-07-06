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


data = io.imread('LaB6_MARCCD.tif')
edges = feature.canny(data, sigma=0.25)
X = np.array(np.where(edges)).T

plt.scatter(X[:, 1], X[:, 0], color='red', s=5)
plt.show()

