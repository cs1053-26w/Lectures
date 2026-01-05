import numpy as np
import imageio

sobel_x = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]], dtype=np.float32)
sobel_y = sobel_x.T

def brightness(img, scale):
    """ Change the brightness of img by scaling its pixel
    values uniformly by scale """
    return np.clip(img * scale, 0, 1)

def threshold(img, thresh):
    """ Threshold img such that the output is
    1 if the input is >= thresh, and 0 otherwise. 
    Pre: img is grayscale """
    return img >= thresh


def mean_filter(img, filter_size):
    """ Apply a square spatial mean filter with side length filter_size
    to a grayscale img. Preconditions:
      - img is a grayscale (2d) float image
      - filter_size is odd """
    H, W = img.shape
    out = np.zeros_like(img)

    hw = filter_size // 2 # half-width

    in_pad = np.pad(img, ((hw, hw), (hw, hw)))
    
    for i in range(H):
        for j in range(W):
            total = 0.0
            for ioff in range(-hw, hw+1):
                for joff in range(-hw, hw+1):
                    total += in_pad[hw + i+ioff, hw + j+joff]
            out[i, j] = total / filter_size**2
    return out

def filter(img, kernel):
    """ Apply filter to img using cross-correlation. Preconditions:
      - img is a grayscale (2d) float image
      - filter is 2d and has odd side lengths """
    H, W = img.shape
    out = np.zeros_like(img)

    # half-width:
    hw = kernel.shape[0] // 2 

    # TODO
   

def convolve(img, kernel):
    """ Apply filter to img using cross-correlation. Preconditions:
      - img is a grayscale (2d) float image
      - filter is square and has odd side length """
    return filter(img, np.fliplr(np.flipud(kernel)))

def grad_mag(img):
    """ Return the gradient magnitude of img as estimated by sobel filters. 
    Pre: img is grayscale. """
    dx = convolve(img, sobel_x)
    dy = convolve(img, sobel_y)
    return np.sqrt(dx ** 2 + dy ** 2)












   
def filter(img, kernel):
    """ Apply filter to img using cross-correlation. Preconditions:
      - img is a grayscale (2d) float image
      - filter is 2d and has odd side lengths """
    H, W = img.shape
    out = np.zeros_like(img)

    # half-widths:
    hwi = kernel.shape[0] // 2 
    hwj = kernel.shape[1] // 2 
    in_pad = np.pad(img, ((hwi, hwi), (hwj, hwj)))
    
    for i in range(H):
        for j in range(W):
            out[i,j] = (kernel * in_pad[hwi+i-hwi:hwi+i+hwi+1, hwj+j-hwj:hwj+j+hwj+1]).sum()
    return out