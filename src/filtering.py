import numpy as np
import imageio

def brightness(img, scale):
    """ Change the brightness of img by scaling its pixel
    values uniformly by scale """
    return np.clip(img * scale, 0, 1)

def threshold(img, thresh):
    """ Threshold img such that the output is
    1 if the input is >= thresh, and 0 otherwise. 
    Pre: img is grayscale """
    return img >= thresh
