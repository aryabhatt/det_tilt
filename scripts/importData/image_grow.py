"""
author: Amanda F (SSRL)

8/4/2017
"""

def grow(img):
    img[:,:-1] += img[:,1:]
    img[:,1:] += img[:,-1:]
    img[:-1,:] += img[1:,:]
    img[1:,:] += img[-1:,:]
    return img