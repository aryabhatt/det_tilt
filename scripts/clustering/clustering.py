"""
author: fangren
"""

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy
from sklearn.metrics.pairwise import pairwise_distances
from scripts.import_image import import_image
import matplotlib.pyplot as plt
import ellipse
import os.path


def clustering(img_path, n_clusters):
    img, im_simple, im_adaptive, im_combine = import_image(img_path)
    data = img
    data[im_combine == 0] = 0
    # decrease size of image for testing
    data = cv2.resize(data, None, fx=0.1, fy=0.1)
    data = data.astype(float)
    print data.shape
    dim_x = data.shape[0]
    dim_y = data.shape[1]
    X_coordinates = np.array(range(data.shape[0])).astype(float)
    Y_coordinates = np.array(range(data.shape[1])).astype(float)
    X_coordinates, Y_coordinates = np.meshgrid(X_coordinates, Y_coordinates)
    keep = data != 0
    data = data[keep]
    X_coordinates = X_coordinates[keep]
    Y_coordinates = Y_coordinates[keep]
    input = np.concatenate(([X_coordinates], [Y_coordinates]), axis=0)
    input = input.T

    # for eps in range(6, 12):
    #     for min_samples in range(5, 11):
    labels = hierarchy.fclusterdata(input, n_clusters, criterion='maxclust',method='single', metric='euclidean')

    # print input.shape
    # print labels.shape
    print np.unique(labels, return_counts= True)
    return labels, X_coordinates, Y_coordinates

if __name__ == '__main__':
    img_path = 'LaB6_MARCCD.tif'
    labels, X_coordinates, Y_coordinates = clustering(img_path, n_clusters=16)
    label_num = 7
    mask = (labels ==label_num)
    save_path = 'plots'

    plt.scatter(X_coordinates, Y_coordinates, c = labels, cmap='nipy_spectral', s = 5)
    #plt.scatter(X_coordinates[mask], Y_coordinates[mask],c = 'k', s=15)
    #plt.show()
    plt.savefig(os.path.join(save_path, 'clustering'))

    # # test ellipse function using one of the arcs, determined by label_num
    label_num = 1
    X = X_coordinates[labels == label_num]
    Y = Y_coordinates[labels == label_num]
    
    p,cost = ellipse.fit_ellipse(X, Y)
    print ((p, cost))
    xx, yy = ellipse.plot_ellipse(p)
    
    plt.scatter(X_coordinates, Y_coordinates, s = 5) # all data points
    plt.scatter(X,Y, c = 'red', s = 10) # data points for fitting
    plt.plot(xx, yy) # plot fit
    plt.axis('equal')
    plt.show()
