"""
author: fangren
"""

import cv2
import matplotlib.pyplot as plt
from scripts.clustering.clustering import clustering

from scripts.fitting import ellipse
from scripts.importData.image_filter import image_filter
from scripts.importData.import_image import import_image

if __name__ == '__main__':
    img = import_image()
    img_edge, im_simple, im_adaptive, im_combine = image_filter(img)
    im_combine = cv2.resize(im_combine, None, fx=0.1, fy=0.1)
    labels, X_coordinates, Y_coordinates = clustering(im_combine, n_clusters=16)

    for label_num in range(16):
        plt.scatter(X_coordinates, Y_coordinates, s=5)  # all data points
    #for label_num in [1,9,11,12,13]:
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
            plt.savefig('..//..//results//fitting//MARCCD//' + str(label_num))
            #plt.savefig('..//..//results//fitting//all' )
            plt.close('all')
        except ValueError:
            plt.scatter(X, Y, c='red', s=10)  # data points for fitting
            plt.savefig('..//..//results//fitting//MARCCD//' + str(label_num))
            plt.close('all')
            continue