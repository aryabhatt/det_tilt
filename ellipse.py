#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, leastsq

def rotMatrix(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.array([[c, -s],[s, c]])

def circle(p, x, y):
    data = np.vstack((x,y))
    
    '''shift origin'''
    pts = data - p[0:2].reshape((2,1)) 

    ''' calc phi '''
    phi = np.arctan2(pts[1,:], pts[0,:])
    csphi = np.vstack((np.cos(phi), np.sin(phi)))

    '''circle'''
    circle = p[:2].reshape((2,1)) + p[2]*csphi
    return np.ravel(data - circle)

def ellipse(p, x, y):
    R = rotMatrix(p[4])
    data = np.vstack((x, y))

    ''' shift origin'''
    pts = data - p[0:2].reshape((2,1)) 

    ''' rotate '''
    pts = np.dot(R.transpose(), pts)

    ''' calc phi '''
    phi = np.arctan2(pts[1,:]/p[3], pts[0,:]/p[2])
    csphi = np.vstack((np.cos(phi), np.sin(phi)))

    ''' ellipse '''
    ell = p[:2].reshape((2,1)) + np.dot(R, p[2:4].reshape((2,1)) * csphi)
    return np.ravel(data - ell) 
    
def fit_circle(x, y, p0 = None):
    if p0 is None:
        p0 = np.zeros(3)
        p0[0] = x.mean()
        p0[1] = y.mean()
        p0[2] = np.sqrt((x - p0[0])**2 + (y-p0[1])**2).mean()

    ff = least_squares(circle, p0, args=(x, y))
    return ff.x, ff.cost

def fit_ellipse(x, y):
    ''' fit as circle to get starting values '''
    p,_ = fit_circle(x, y)

    ''' start from circle to fit an ellipse '''
    p0 = np.array([p[0], p[1], p[2], p[2]/2, 0.])
    ff = least_squares(ellipse, p0, args=(x, y), method='lm')
    return ff.x, ff.cost

def plot_ellipse(p, min_phi = 0., max_phi = 6.242):
    phi = np.linspace(min_phi, max_phi, 100)
    csphi = np.vstack((np.cos(phi), np.sin(phi)))
    pts = p[2:4].reshape((2,1))*csphi
    R = rotMatrix(p[4])
    pts = p[:2].reshape((2,1)) + np.dot(R,  pts)
    return pts[0,:], pts[1,:] 

def plot_circle(p, min_phi = 0., max_phi = 6.242):
    phi = np.linspace(min_phi, max_phi, 100)
    csphi = np.vstack((np.cos(phi), np.sin(phi)))
    pts = p[2] * csphi + p[:2].reshape((2,1))
    return pts[0,:], pts[1,:]

def circle3pt(x, y):
    if not x.size == 3 or not y.size == 3:
        raise ValueError

    '''
    x^2 + y^2 + a x + b y + c = 0
    a x + c y + c = -(x^2 + y^2)
    [ X   Y  1 ] {A} =   B
    '''
    fx = lambda x, y: -(x**2 + y**2)
    X = [[ x[0], y[0], 1], [x[1], y[1], 1], [x[2], y[2], 1]]
    B = [ -fx(x[0], y[0]), -fx(x[1], y[1]), -fx(x[2], y[2]) ] 
    A = np.linalg.linsolve(X, B)
    return  A[0]/2., A[1]/2., np.sqrt(A[2])

if __name__ == '__main__':
    np.random.seed(100)
    th = np.linspace(np.pi, 2*np.pi, 100)
    a = 30
    b = 20
    cx = 5
    cy = 8
    alpha = np.deg2rad(30.)
    p = np.array([cx, cy, a, b, alpha])
    x = a * np.cos(th)
    y = b * np.sin(th)
    R = rotMatrix(alpha)
    pts = np.dot(R, np.vstack((x, y))) + np.array([[cx], [cy]])
    pts += np.random.randn(2, 100)*1
    irr = np.random.permutation(100)
    pts = pts[:,irr]

    p,cost = fit_ellipse(pts[0,:], pts[1,:])
    print ((p, cost))
    r,s = plot_ellipse(p)
    
    plt.scatter(pts[0,:], pts[1,:])
    plt.scatter(r, s)
    plt.axis('equal')
    plt.show()
