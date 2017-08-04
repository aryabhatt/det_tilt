"""
author: ronpandolfi, fangren
"""
import matplotlib.pyplot as plt
import numpy as np
import pyFAI
from pyFAI import calibrant

from scripts.importData.circular_mask import circular_mask
from scripts.importData.import_image import import_image
from scripts.parsing_calib import parse_calib


#print calibrant.calibrant_factory()



def create_fake_image(params, pixelSize = 79, shape = (2048, 2048)):
    detector_dist, detect_tilt_alpha, detect_tilt_delta, bcenter_x, bcenter_y, wavelength = params
    # transfer WxDiff parameters to pyFAI parameters
    Rot = (np.pi * 2 - detect_tilt_alpha) / (2 * np.pi) * 360  # detector rotation
    tilt = detect_tilt_delta / (2 * np.pi) * 360  # detector tilt  # wavelength
    d = detector_dist * pixelSize * 0.001  # measured in milimeters
    wavelength = wavelength* 1e-10
    # create ai object
    ai = pyFAI.AzimuthalIntegrator(wavelength= wavelength)
    ai.setFit2D(d, bcenter_x, bcenter_y, tilt, Rot, pixelSize, pixelSize)
    c = calibrant.ALL_CALIBRANTS['LaB6']
    c.set_wavelength(ai.wavelength)
    imArray = c.fake_calibration_image(ai, shape=shape, Imax=255, U=0, V=0, W=0.00005)
    mask = circular_mask(shape, (shape[0]/2, shape[1]/2), shape[0]/2)
    #print imArray.shape, mask.shape
    imArray[mask] = 0
    return imArray

if __name__ == '__main__':
    # import calibration parameters for the real image
    params = parse_calib('LaB6//3.calib')
    shape = (2048, 2048)
    pixelSize = 79
    # create fake image
    imArray = create_fake_image(params, pixelSize, shape)
    #print np.unique(imArray, return_counts= True)
    # import real image
    img, im_simple, im_adaptive, im_combine = import_image('LaB6//3.tif')
    im_combine = im_combine.astype(float)
    #print np.unique(im_combine, return_counts= True)
    # visualization
    plt.imshow(imArray, alpha=0.7)
    plt.imshow(im_combine, alpha=0.7)
    plt.show()