"""
author: fangren
"""
import numpy as np

def circular_mask ((n1, n2), (a, b), radius):
    y, x = np.ogrid[-a:n1 - a, -b:n2 - b]
    mask = x * x + y * y >= radius * radius
    return mask
