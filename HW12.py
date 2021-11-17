import matplotlib as plt
import cv2
import numpy as np
import math
#import PyQt5
from cfg import *

# gaussian blur
img1 = cv2.imread(q2_folder + 'Lenna_whiteNoise.jpg')
cv2.imshow('Lenna with whiteNoise', img1)
cv2.waitKey(0)
blur1 = cv2.GaussianBlur(img1, (5, 5), 0)
cv2.imshow('Lenna after gaussian', blur1)
cv2.waitKey(0)

# Bilateral Filter
blur2 = cv2.bilateralFilter(img1, 9, 90, 90)  # unsure arg
cv2.imshow('Lenna after Bilateral', blur2)
cv2.waitKey(0)

# median filter
blur3 = cv2.medianBlur(img1, 5)
cv2.imshow('Lenna after median5*5', blur3)
cv2.waitKey(0)
blur4 = cv2.medianBlur(img1, 3)
cv2.imshow('Lenna after median3*3', blur4)
cv2.waitKey(0)

# **************************************************************************
# HW3


def normalize(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def Gaus_filter(size, sigma=1, verbose=False):
    array_1d = np.empty(size)
    start = size // 2
    # initialized(好像可用linspace取代)
    for index in range(size):
        array_1d[index] = index - start
    # normalized
    for index in range(size):
        array_1d[index] = normalize(array_1d[index], 0, sigma)
    array_2d = np.outer(array_1d.T, array_1d.T)
    array_2d *= 1.0 / array_2d.max()

    return array_2d  # return filter

# make sure convert to grayscale before calling convolution


def convolution(image, filter, avg=False, verbose=False):
    img_rows, img_cols = image.shape
    fil_row, fil_col = filter.shape
    result = np.zeros((img_rows, img_cols))  # for output

    # padding block
    padding = int((fil_row - 1) / 2)
    pad_img = np.zeros((padding*2 + img_rows, padding*2 + img_cols))
    pad_img[padding: -padding, padding: -padding] = image
    # conv operation
    for row in range(img_rows):
        for col in range(img_rows):
            # filter 走到之處九宮格相乘再全sum起來
            result[row, col] = np.sum(
                filter * pad_img[row: row + fil_row, col: col + fil_col])
            if avg:
                # divide total pixels number
                result[row, col] /= filter.shape[0] * filter.shape[1]
    return result


def gaussian_blur(img, filter_size, verbose=False):
    sigma1 = math.sqrt(filter_size)
    filter = Gaus_filter(filter_size, sigma1, verbose=verbose)
    return convolution(img, filter, avg=True, verbose=verbose)


gau2 = cv2.imread(q3_folder + 'House.jpg')
gau2 = cv2.cvtColor(gau2, cv2.COLOR_BGR2GRAY)

cv2.imshow('after hand made gaussian_filter', gaussian_blur(gau2, 5))
cv2.waitKey(0)
