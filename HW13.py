import matplotlib as plt
import cv2
import numpy as np
import math
# import PyQt5
from cfg import *
# **************************************************************************
# HW3


# sqrt() ->square root
def normalize(x, m, seed):
    return 1 / (np.sqrt(np.pi * 2) * seed) * np.e ** (-np.power((x - m) / seed, 2) / 2)


def Gaus_filter(size, sigma=1):
    array_1d = np.empty(size)
    start = size // 2
    # initialized(好像可用linspace取代)
    for index in range(size):
        array_1d[index] = index - start
    print(array_1d)
    # normalized
    for index in range(size):
        array_1d[index] = normalize(array_1d[index], 0, sigma)
    array_2d = np.outer(array_1d.T, array_1d.T)
    array_2d *= 1.0 / array_2d.max()

    print(array_2d)
    return array_2d  # return filter

# make sure convert to grayscale before calling convolution


def convolution(image, filter, avg=False):

    img_rows, img_cols = image.shape
    fil_row, fil_col = filter.shape
    result = np.zeros((img_rows, img_cols))  # for output

    # padding block
    padding_r = int((fil_row - 1) / 2)
    padding_c = int((fil_col - 1) / 2)
    pad_img = np.zeros(((padding_r*2) + img_rows, (padding_c*2) + img_cols))
    pad_img[padding_r:pad_img.shape[0] - padding_r,
            padding_c:pad_img.shape[1] - padding_c] = image
    # conv operation
    for row in range(img_rows):
        for col in range(img_cols):
            # filter 走到之處九宮格相乘再全sum起來
            result[row, col] = np.sum(
                filter * pad_img[row: row + fil_row, col: col + fil_col])
            if avg:
                # divide total pixels number
                result[row, col] /= filter.shape[0] * filter.shape[1]
    return result


def gaussian_blur(img, filter_size):
    sigma1 = math.sqrt(filter_size)  # square root the filter_size
    filter = Gaus_filter(filter_size,  math.sqrt(filter_size))
    return convolution(img, filter, avg=True)


gau2 = cv2.imread(q3_folder + 'House.jpg')
gau2 = cv2.cvtColor(gau2, cv2.COLOR_BGR2GRAY)
cv2.imshow('House', gau2)
cv2.waitKey(0)

cv2.imwrite("./data/House2.jpg", gaussian_blur(gau2, 3))
House2 = cv2.imread('./data/House2.jpg')
cv2.imshow('after hand made gaussian_filter', House2)
cv2.waitKey(0)

# ***********************************************************************************
# Sobel X
sobFilterx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobFiltery = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobelx_result = convolution(gaussian_blur(gau2, 3), sobFilterx)
sobely_result = convolution(gaussian_blur(gau2, 3), sobFiltery)
sobelx_result *= 255.0 / sobelx_result.max()
'''
為啥直接show會超怪
cv2.imshow('after diy sobelx', sobelx_result)
cv2.waitKey(0)
'''
cv2.imwrite('sobel2.jpg', sobelx_result)


'''
sobelmmm_result = convolution(cv2.GaussianBlur(gau2, (3, 3), 0), sobFilterx)
cv2.imshow('after hand ', sobelmmm_result)
cv2.waitKey(0)

xy_magnitude = np.sqrt(np.square(sobelx_result) + np.square(sobely_result))
# normalize each pixel in between 0 ~ 255
xy_magnitude *= 255/xy_magnitude.max()
#cv2.imshow('after hand made sobel_x sobel_y', xy_magnitude)
# cv2.waitKey(0)
'''