#! /usr/local/bin/python

import numpy as np
from skimage import io
from skimage import feature
from skimage.feature import blob_log
from scipy.spatial import cKDTree
import ellipse
import fabio

import sys
import matplotlib.pyplot as plt


def clusterPts(pts, radius=25, num_clusters=5):
    kdtree = cKDTree(pts, leafsize=20) 
    nPts = pts.shape[0]
   
    #idx = np.unique(np.random.randint(0, nPts, 40)[::-1])
    #seeds = pts[idx,:].tolist()
    #seeds = [[520, 1886], [330, 1764], [329, 1510]] 
    clusters = []
    for j in range(nPts):
        ''' skip we aleady have this cluster '''
        skip = False
        for cl in clusters:
            if j in cl:
                skip = True
                break
        if skip: continue
        pt = kdtree.data[j,:]

        ''' stop, if have required num of clusters '''
        if len(clusters) == num_clusters:
            break

        arc = set()
        while True:
            idx = kdtree.query_ball_point(pt, radius)
            i = idx[-1] 
            if arc.issuperset(idx):
                break
            arc = arc.union(idx)
            pt = kdtree.data[i,:]
        new = True
        for i in range(len(clusters)):
            tmp = set(clusters[i])
            if len(tmp.intersection(arc)) > 0:
                clusters[i] = list(tmp.union(arc))
                new = False
        if new: 
            clusters.append(list(arc))
  
    return clusters
        
if __name__ == '__main__':
    data = fabio.open('LaB6_MARCCD.tif').data
    # edges = feature.canny(data, sigma=0.25)
    # y,x = np.where(edges)

    blobs_log = blob_log(data, max_sigma=3, num_sigma=3, threshold=.1)
    y,x,sigma = blobs_log.T
    plt.show()

    pts = np.vstack((x, y)).transpose()

    # cluster points
    radius = 100
    import time
    t0 = time.time()
    labels = clusterPts(pts, radius, num_clusters=10)
    t1 = time.time() - t0

    for lbl in labels:
        ell = pts[lbl,:].astype(float)
        shift = ell.min(axis=0)
        scale = ell.max()
        pp = (ell - shift)/scale
        if len(pp)>3:
            p,_ = ellipse.fit_ellipse(pp[:,0], pp[:,1])
            xx, yy = ellipse.plot_ellipse(p)
            xx = xx * scale + shift[0]
            yy = yy * scale + shift[1]
            plt.imshow(data)
            plt.scatter(ell[:,0], ell[:,1])
            plt.plot(xx, yy)
            plt.show()
    print ('num of clusters = %d' % len(labels))
    print ('time taken = %f' % t1)
