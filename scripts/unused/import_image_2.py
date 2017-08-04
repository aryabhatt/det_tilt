#! /usr/local/bin/python

import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
from skimage import io

data = io.imread('..//..//LaB6//LaB6_MARCCD.tif')
edges = feature.canny(data, sigma=0.25)
print edges
plt.imshow(edges)
coordinates = np.array(np.where(edges)).T
print coordinates[:, 1]
print coordinates[:, 0]
#plt.scatter(coordinates[:, 1], coordinates[:, 0], color='red', s=5)
plt.show()

