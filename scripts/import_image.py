"""
Created on Wed Jun 28 09:37:11 2017

@author: Alex Belianinov

Adaptive thresholding on xray data for Fang Ren

Editted by Fang Ren
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def import_image(image_path):
    img = cv2.imread(image_path, 0)
    im_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, im_simple = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    im_combine = (im_adaptive + im_simple) / 2
    im_combine[im_combine > 0] = 255
    return (img,
     im_simple,
     im_adaptive,
     im_combine)


if __name__ == '__main__':
    img, im_simple, im_adaptive, im_combine = import_image('LaB6_MARCCD.tif')
    print np.unique(im_simple, return_counts=True)
    print np.unique(im_adaptive, return_counts=True)
    print np.unique(im_combine, return_counts=True)
    plt.figure(1, figsize=(9, 9))
    plt.subplot(221)
    plt.title('raw image')
    plt.imshow(img)
    plt.clim(img.min(), (img.max() - img.min()) * 0.1 + img.min())
    plt.subplot(222)
    plt.title('adaptive thresholding image')
    plt.imshow(im_adaptive)
    plt.subplot(223)
    plt.title('simple thresholding image')
    plt.imshow(im_simple)
    plt.subplot(224)
    plt.title('combine adaptive and simple thresholding')
    (plt.imshow(im_combine), plt.show())
    plt.savefig('plots\\1_comparison.jpeg')
    cv2.imwrite('plots\\1_result.jpeg', im_combine)