"""
author: fangren
"""

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from import_image import import_image
import matplotlib.pyplot as plt
import ellipse
import os.path
img, im_simple, im_adaptive, im_combine = import_image('LaB6_MARCCD.tif')
data = img
data[im_combine == 0] = 0
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
save_path = 'plots//db_test'
# for eps in range(6, 12):
#     for min_samples in range(5, 11):

eps = 8
min_samples = 5
db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm='brute')
db.fit(input)
labels = db.labels_
# print labels
# mask = labels != -1
# plt.scatter(X_coordinates[mask], Y_coordinates[mask], c=labels[mask], cmap='Paired', s=5)
# plt.savefig(os.path.join(save_path, 'eps=' + str(eps) + ';min_samples=' + str(min_samples)))

# test ellipse function
label_num = 14
p,cost = ellipse.fit_ellipse(X_coordinates[labels == label_num], Y_coordinates[labels == label_num])
print ((p, cost))
r,s = ellipse.plot_ellipse(p)

plt.scatter(X_coordinates, Y_coordinates)
plt.scatter(X_coordinates[labels == label_num], Y_coordinates[labels == label_num], c = 'red')
plt.scatter(r, s)
plt.xlim(0, dim_x)
plt.ylim(0,dim_y)
plt.show()
