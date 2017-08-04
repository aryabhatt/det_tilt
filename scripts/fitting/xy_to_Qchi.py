# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:14:05 2016

@author: fangren

Qsum function modified by Ronald Pandolf
"""

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from scripts.importData.import_image import import_image
from scripts.importData.parsing_calib import parse_calib


def transRev(xdet, ydet, x0, y0):
    """
    translation from detector coordinates (detector corner as the origin)
    into detector plane coordinates (beam center as the new origin)
    x0, y0 are coordiantes of the beam center in  the old coordinate system
    xdet, ydet are pixel coordinates on the detector coordinate system
    return new coordiantes x, y in the new coordiante system
    """
    x = xdet - x0
    y = ydet - y0
    return x, y


def rotRev(Rot, x, y):
    """
    Rotation according to the beam center (0, 0), in the detector plane
    """
    xbeam = x*np.cos(-Rot) - y*np.sin(-Rot)
    ybeam = x*np.sin(-Rot) + y*np.cos(-Rot)
    return xbeam, ybeam


def calTheta_chi(d, tilt, xbeam, ybeam):
    """
    calculate theta angle from the detector distance d (along beam travel
    direction), and tilting angle of the detector
    return theta and chi angles
    """
    p1 = np.sqrt(d**2+xbeam**2-2*xbeam*d*np.sin(tilt))
    gama = np.arccos((d**2+p1**2-xbeam**2)/(2*d*p1))
    x = d*np.tan(gama)
    p2 = d/np.cos(gama)
    y = ybeam*p2/p1
    rSqr = x**2 + y**2
    twoTheta = np.arctan(np.sqrt(rSqr/(d**2)))
    chi = np.arctan(y/x)
    return twoTheta/2, chi


def calQ(lamda, theta):
    """
    calculate Q from theta angle and beam energy lamda
    """
    return 4*np.pi*np.sin(theta)/lamda
    

def Qchi(xdet, ydet, params):
    """
    Integrate the four functions: transRev, rotRev, calTheta_chi, calQ
    return Q and chi
    """
    detector_dist, detect_tilt_alpha, detect_tilt_delta, bcenter_x, bcenter_y, wavelength = params
    x, y = transRev(xdet, ydet, bcenter_x, bcenter_y)
    xbeam, ybeam = rotRev(detect_tilt_alpha, x, y)
    theta, chi = calTheta_chi(detector_dist, detect_tilt_delta, xbeam, ybeam)
    Q = calQ(wavelength, theta)
    return Q, chi


def polarCorr(intensity, Q, chi, lamda, PP):
    """
    polarization correction
    """
    Qx = Q*np.sin(chi)
    Qy = Q*np.cos(chi)
    thetax = np.arcsin(Qx * lamda / (4 * np.pi ))
    thetay = np.arcsin(Qy * lamda / (4 * np.pi ))
    PCorr = 0.5 * ((1-PP)*np.cos(thetay)+PP*np.cos(thetax)+1) 
    return intensity / PCorr


if __name__ == '__main__':
    # open MARCCD tiff image
    img, im_simple, im_adaptive, im_combine = import_image('LaB6//3.tif')
    # import calibration parameters for the real image
    params = parse_calib('LaB6//3.calib')
    detector_dist, detect_tilt_alpha, detect_tilt_delta, bcenter_x, bcenter_y, wavelength  = params
    PP = 0.95   # beam polarization, decided by beamline setup

    # visualize image
    s1 = img.shape[0]
    s2 = img.shape[1]
    X = [i + 1 for i in range(s1)]
    Y = [i + 1 for i in range(s2)]
    X, Y = np.meshgrid(X, Y)

    plt.figure(1)
    plt.subplot(211)
    plt.pcolormesh(X, Y, img)
    #plt.imshow(img)

    # Calculate Q, chi, polarized corrected intensity(Ipol) arrays

    Q = np.zeros((s1,s2))
    chi = np.zeros((s1,s2))
    Ipol = np.zeros((s1,s2))
    Q[:], chi[:] = Qchi(X, Y, params)
    Ipol = polarCorr(img, Q, chi, wavelength, PP)

    # generate a Q-chi plot with polarization correction in log scale
    plt.subplot(212)
    plt.title('Q-chi_polarization corrected.tif')
    plt.pcolormesh(Q, chi, np.log(np.ma.masked_where(img == 0, Ipol)))
    plt.colorbar()
    plt.xlabel('Q')
    plt.ylabel('chi')
    plt.show()

