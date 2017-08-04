"""
author: fangren
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
import os
from scripts.importData.image_filter import image_filter
from scripts.importData.import_image import import_image


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
    save_path = '..//..//results//clustering'
    plt.scatter(X_coordinates, Y_coordinates, c = labels, cmap='nipy_spectral', s = 5)
    #plt.scatter(X_coordinates[mask], Y_coordinates[mask],c = 'k', s=15)
    #plt.show()
    plt.savefig(os.path.join(save_path, 'clustering'))
    plt.close('all')

    img = import_image('..//..//Lab6//1.tif')
    img_edge, im_simple, im_adaptive, im_combine = image_filter(img)
    im_combine = cv2.resize(im_combine, None, fx=0.1, fy=0.1)
    labels, X_coordinates, Y_coordinates = clustering(im_combine, n_clusters=16)
    label_num = 7
    mask = (labels ==label_num)
    save_path = '..//..//results//clustering'
    plt.scatter(X_coordinates, Y_coordinates, c = labels, cmap='nipy_spectral', s = 5)
    #plt.scatter(X_coordinates[mask], Y_coordinates[mask],c = 'k', s=15)
    #plt.show()
    plt.savefig(os.path.join(save_path, 'clustering1'))
    plt.close('all')

    img = import_image('..//..//Lab6//2.tif')
    img_edge, im_simple, im_adaptive, im_combine = image_filter(img)
    im_combine = cv2.resize(im_combine, None, fx=0.1, fy=0.1)
    labels, X_coordinates, Y_coordinates = clustering(im_combine, n_clusters=16)
    label_num = 7
    mask = (labels ==label_num)
    save_path = '..//..//results//clustering'
    plt.scatter(X_coordinates, Y_coordinates, c = labels, cmap='nipy_spectral', s = 5)
    #plt.scatter(X_coordinates[mask], Y_coordinates[mask],c = 'k', s=15)
    #plt.show()
    plt.savefig(os.path.join(save_path, 'clustering2'))
    plt.close('all')

    img = import_image('..//..//Lab6//3.tif')
    img_edge, im_simple, im_adaptive, im_combine = image_filter(img)
    im_combine = cv2.resize(im_combine, None, fx=0.1, fy=0.1)
    labels, X_coordinates, Y_coordinates = clustering(im_combine, n_clusters=16)
    label_num = 7
    mask = (labels ==label_num)
    save_path = '..//..//results//clustering'
    plt.scatter(X_coordinates, Y_coordinates, c = labels, cmap='nipy_spectral', s = 5)
    #plt.scatter(X_coordinates[mask], Y_coordinates[mask],c = 'k', s=15)
    #plt.show()
    plt.savefig(os.path.join(save_path, 'clustering3'))
    plt.close('all')