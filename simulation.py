"""
author: ronpandolfi, fangren
"""
import pyFAI
from pyFAI import calibrant
import matplotlib.pyplot as plt
import numpy as np
from circular_mask import circular_mask
from import_image import import_image
from parsing_calib import parse_calib

#print calibrant.calibrant_factory()

detector_dist, detect_tilt_alpha, detect_tilt_delta, wavelength, bcenter_x, bcenter_y = parse_calib('LaB6//3.calib')
pixelSize = 79
shape = (2048, 2048)

#
Rot = (np.pi * 2 - detect_tilt_alpha) / (2 * np.pi) * 360  # detector rotation
tilt = detect_tilt_delta / (2 * np.pi) * 360  # detector tilt  # wavelength
d = detector_dist * pixelSize * 0.001  # measured in milimeters

ai = pyFAI.AzimuthalIntegrator(wavelength= wavelength * 1e-10)
ai.setFit2D(d, bcenter_x, bcenter_y, tilt, Rot, pixelSize, pixelSize)

c = calibrant.ALL_CALIBRANTS['LaB6']
c.set_wavelength(ai.wavelength)
imArray = c.fake_calibration_image(ai, shape=shape, Imax=255, U=0, V=0, W=0.00005)
mask = circular_mask(shape, (shape[0]/2, shape[1]/2), shape[0]/2)

print imArray.shape, mask.shape
imArray[mask] = 0
plt.imshow(imArray, alpha=0.7)
print np.unique(imArray, return_counts= True)
# import real image
img, im_simple, im_adaptive, im_combine = import_image('LaB6//3.tif')
im_combine = im_combine.astype(float)
print np.unique(im_combine, return_counts= True)
# im_combine[im_combine == 0] = np.nan

plt.imshow(im_combine, alpha=0.7)
plt.show()