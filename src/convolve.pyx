cimport numpy as cnp
from skimage.exposure import rescale_intensity
import numpy as np
import cv2 as cv

def convolution(cnp.ndarray img, cnp.ndarray kernel):
    cdef int kernel_h, kernel_w, img_h, img_w, padding, h, w, conv_value
    cdef cnp.ndarray result, section

    # kernel height and width
    kernel_h, kernel_w = kernel.shape[0], kernel.shape[1]
    # image height and width
    img_h, img_w = img.shape[0], img.shape[1]
    # initialize the result of the convolution
    result = np.zeros((img_h, img_w), np.float32)
    # padding with replication
    padding = (kernel_h - 1) // 2
    img = cv.copyMakeBorder(img, padding,padding,padding,padding, cv.BORDER_REPLICATE)
    # convolution operation
    for h in range(padding, img_h+padding):
        for w in range(padding, img_w+padding):
            section = img[h-padding:h+padding+1, w-padding:w+padding+1]
            conv_value = (section * kernel).sum()
            result[h-padding, w-padding] = conv_value

    # rescale to required pixel range
    result = rescale_intensity(result, in_range=(0,255))
    result = (result*255).astype('uint8')

    return result