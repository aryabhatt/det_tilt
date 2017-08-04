"""
Created on Wed Jun 28 09:37:11 2017

@author: Alex Belianinov, Fang Ren

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from import_image import import_image
from skimage import feature
from image_grow import grow

def image_filter(img):
    # edge finding:
    im_edge = feature.canny(img, sigma=0.25)
    # simple thresholding
    ret, im_simple = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    # adaptive thresholding
    im_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    im_combine = (im_edge + im_adaptive + im_simple) / 3
    im_combine = grow(im_combine)
    im_combine[im_combine > 0] = 255
    return im_edge, im_simple, im_adaptive, im_combine

if __name__ == '__main__':
    img = import_image()
    im_edge, im_simple, im_adaptive, im_combine = image_filter(img)
    print np.unique(im_simple, return_counts=True)
    print np.unique(im_adaptive, return_counts=True)
    print np.unique(im_combine, return_counts=True)
    plt.figure(1, figsize=(9, 9))
    plt.subplot(221)
    plt.title('edge finding')
    plt.imshow(im_edge)
    plt.subplot(222)
    plt.title('adaptive thresholding image')
    plt.imshow(im_adaptive)
    plt.subplot(223)
    plt.title('simple thresholding image')
    plt.imshow(im_simple)
    plt.subplot(224)
    plt.title('combine edge finding, adaptive and simple thresholding')
    plt.imshow(im_combine)
    plt.show()
    # plt.savefig('plots\\1_comparison.jpeg')
    # cv2.imwrite('plots\\1_result.jpeg', im_combine)