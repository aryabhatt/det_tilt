"""
author: fangren
"""

import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

from scripts.importData.import_image import import_image


def clustering(img_path, eps, min_samples):
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
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm='brute')
    db.fit(input)
    labels = db.labels_
    return labels, X_coordinates, Y_coordinates, dim_x, dim_y

if __name__ == '__main__':
    eps = 8
    min_samples = 5
    labels, X_coordinates, Y_coordinates, dim_x, dim_y = clustering(img_path = 'LaB6_MARCCD.tif', eps = eps, min_samples = min_samples)
    print labels
    mask = labels != -1
    save_path = 'plots//db_test'
    plt.scatter(X_coordinates[mask], Y_coordinates[mask], c=labels[mask], cmap='Paired', s=5)
    plt.show()
    plt.savefig(os.path.join(save_path, 'eps=' + str(eps) + ';min_samples=' + str(min_samples)))

    # # test ellipse function using one of the arcs, determined by label_num
    # label_num = 14
    # X = X_coordinates[labels == label_num]
    # x_max = X.max()
    # x_min = X.min()
    # X = (X-x_min)/x_max
    #
    # Y = Y_coordinates[labels == label_num]
    # Y = (Y-x_min)/x_max
    #
    # print X, Y
    #
    # p,cost = ellipse.fit_ellipse(X, Y)
    # print ((p, cost))
    # xx, yy = ellipse.plot_ellipse(p)
    #
    # # scale back
    # xx = xx * x_max + x_min
    # X = X  * x_max + x_min
    # yy = xx * x_max + x_min
    # Y = Y * x_max + x_min
    #
    # plt.scatter(X_coordinates, Y_coordinates, s = 5) # all data points
    # plt.scatter(X,Y, c = 'red', s = 5) # data points for fitting
    # plt.scatter(xx, yy, c = 'orange', s = 5) # plot fit
    # plt.xlim(0, dim_x)
    # plt.ylim(0,dim_y)
    # plt.show()
