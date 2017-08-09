"""
Created on Wed Jun 28 09:37:11 2017

@author: Alex Belianinov

Adaptive thresholding on xray data for Fang Ren

Editted by Fang Ren
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def import_image(image_path = '..//..//LaB6//MARCCD//1.tif'):
    img = cv2.imread(image_path, 0)
    return img

if __name__ == '__main__':
    img = import_image()
    plt.figure(1, figsize=(9, 9))
    plt.title('raw image')
    print img.shape
    plt.imshow(img)
    plt.clim(img.min(),  (img.max() - img.min()) * 0.05 + img.min())
    plt.show()