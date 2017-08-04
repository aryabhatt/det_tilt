"""
author: fangren
"""

import os.path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from scripts import ellipse
from scripts.importData.import_image import import_image
from scripts.importData.image_filter import image_filter

def clustering(data, n_clusters):
    # decrease size of image for testing
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
    img = import_image()
    img_edge, im_simple, im_adaptive, im_combine = image_filter(img)
    im_combine = cv2.resize(im_combine, None, fx=0.1, fy=0.1)
    labels, X_coordinates, Y_coordinates = clustering(im_combine, n_clusters=16)
    label_num = 7
    mask = (labels ==label_num)
    save_path = 'plots'

    #plt.scatter(X_coordinates, Y_coordinates, c = labels, cmap='nipy_spectral', s = 5)
    #plt.scatter(X_coordinates[mask], Y_coordinates[mask],c = 'k', s=15)
    #plt.show()
    #plt.savefig(os.path.join(save_path, 'clustering'))

    # # test ellipse function using one of the arcs, determined by label_num
    # for label_num in range(16):
    plt.scatter(X_coordinates, Y_coordinates, s=5)  # all data points
    for label_num in [1,9,11,12,13]:
        try:
            X = X_coordinates[labels == label_num]
            Y = Y_coordinates[labels == label_num]

            p,cost = ellipse.fit_ellipse(X, Y)
            print ((p, cost))
            xx, yy = ellipse.plot_ellipse(p)
            plt.scatter(X,Y, c = 'red', s = 10) # data points for fitting
            plt.plot(xx, yy) # plot fit
            plt.axis('equal')
            #plt.show()
            #plt.savefig('..//..//results//fitting//' + str(label_num))
            plt.savefig('..//..//results//fitting//all' )
            #plt.close('all')
        except ValueError:
            continue