#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from cfg import *
root = tk.Tk()
root.title('E94081107 opencv HW1')
root.geometry('720x360')


def button_event1():
    # hw1-1
    img1_path = q1_folder + 'Sun.jpg'
    img1 = cv2.imread(img1_path)
    cv2.imshow('Sun.jpg', img1)
    print('Sun_width : ', img1.shape[1])
    print('Sun_height :', img1.shape[0])
    cv2.waitKey(0)
    cv2.destroyWindow('Sun.jpg')


def button_event2():

    # hw1-2
    img1_path = q1_folder + 'Sun.jpg'
    img1 = cv2.imread(img1_path)
    blue, green, red = cv2.split(img1)
    zeros = np.zeros(blue.shape[:2], dtype='uint8')
    cv2.imshow('B channel', cv2.merge([blue, zeros, zeros]))
    cv2.imshow('G channel', cv2.merge([zeros, green, zeros]))
    cv2.imshow('R channel', cv2.merge([zeros, zeros, red]))
    cv2.waitKey(0)
    cv2.destroyWindow('B channel')
    cv2.destroyWindow('G channel')
    cv2.destroyWindow('R channel')


def button_event3():
    img1_path = q1_folder + 'Sun.jpg'
    img1 = cv2.imread(img1_path)
    # hw 1-3
    merge1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    print(merge1.shape)
    cv2.imshow('merge1', merge1)
    cv2.imshow('merge!!!!', cv2.merge([merge1, merge1, merge1]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def button_event4():
    alpha1 = 0.5
    src11 = cv2.imread(q1_folder + 'Dog_Weak.jpg')
    src22 = cv2.imread(q1_folder + 'Dog_Strong.jpg')
    blend3 = cv2.addWeighted(src11, alpha1, src22, (1-alpha1), 0)
    cv2.imshow('dogs blend(press q to exit)', blend3)

    def trackbar(a):
        global blend3
        print(a)
        alpha1 = a * (1/255)
        blend3 = cv2.addWeighted(src11, alpha1, src22, (1-alpha1), 0)
        cv2.imshow('dogs blend(press q to exit)', blend3)
        # blend3 = np.uint8(np.clip(src11 * alpha1 + src2 * (1 - alpha1) + 0), 0, 255)

    cv2.namedWindow('dogs blend(press q to exit)')
    cv2.createTrackbar(
        'Blend', 'dogs blend(press q to exit)', 0, 255,  trackbar)
    cv2.setTrackbarPos('Blend', 'dogs blend(press q to exit)', 255)
    """while(True):
        cv2.imshow('dogs blend(press q to exit)', blend3)
        if(cv2.waitKey(1) == ord('q')):  # (press q to exit)
            break"""
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def button_event21():
    # gaussian blur
    img1 = cv2.imread(q2_folder + 'Lenna_whiteNoise.jpg')
    cv2.imshow('Lenna with whiteNoise', img1)
    cv2.waitKey(0)
    blur1 = cv2.GaussianBlur(img1, (5, 5), 0)
    cv2.imshow('Lenna after gaussian', blur1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def button_event22():
    # Bilateral Filter
    img1 = cv2.imread(q2_folder + 'Lenna_whiteNoise.jpg')
    cv2.imshow('Lenna with whiteNoise', img1)
    cv2.waitKey(0)
    blur2 = cv2.bilateralFilter(img1, 9, 90, 90)  # unsure arg
    cv2.imshow('Lenna after Bilateral', blur2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def button_event23():
    # median filter
    img1 = cv2.imread(q2_folder + 'Lenna_whiteNoise.jpg')
    cv2.imshow('Lenna with whiteNoise', img1)
    cv2.waitKey(0)
    blur3 = cv2.medianBlur(img1, 5)
    cv2.imshow('Lenna after median5*5', blur3)
    cv2.waitKey(0)
    blur4 = cv2.medianBlur(img1, 3)
    cv2.imshow('Lenna after median3*3', blur4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normalize(x, m, seed):
    return 1 / (np.sqrt(np.pi * 2) * seed) * np.e ** (-np.power((x - m) / seed, 2) / 2)


def Gaus_filter(size, sigma=1):
    array_1d = np.empty(size)
    start = size // 2
    # initialized(好像可用linspace取代)
    for index in range(size):
        array_1d[index] = index - start
    # print(array_1d)
    # normalized
    for index in range(size):
        array_1d[index] = normalize(array_1d[index], 0, sigma)
    array_2d = np.outer(array_1d.T, array_1d.T)
    array_2d *= 1.0 / array_2d.max()

    # print(array_2d)
    return array_2d  # return filter

# make sure convert to grayscale before calling convolution

# Q3's functions******************************************************************************


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
    # filter = Gaus_filter(filter_size,  math.sqrt(filter_size))
    filter_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filter_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    g_filter = np.exp(-(filter_x ** 2 + filter_y**2))
    g_filter *= 1/g_filter.sum()
    img = cv2.filter2D(src=img, kernel=g_filter, ddepth=-1)
    return img


sobFilterx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobFiltery = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
# *************************************************************************************************


def button_event31():  # sobel x
    gau2 = cv2.imread(q3_folder + 'House.jpg')
    gau2 = cv2.cvtColor(gau2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('House', gau2)
    cv2.waitKey(0)
    cv2.imshow('after hand made gaussian_filter', gaussian_blur(gau2, 3))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def button_event32():  # sobel y
    # Sobel X
    global sobFilterx, sobFiltery
    gau2 = cv2.imread(q3_folder + 'House.jpg')
    gau2 = cv2.cvtColor(gau2, cv2.COLOR_BGR2GRAY)
    sobelx_result = convolution(gaussian_blur(gau2, 3), sobFilterx)
    cv2.imwrite('./data/sobelx.jpg', sobelx_result)
    sobelx_result = cv2.imread('./data/sobelx.jpg')
    cv2.imshow('after diy sobelx', sobelx_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def button_event33():  # sobel xy
    global sobFilterx, sobFiltery
    gau2 = cv2.imread(q3_folder + 'House.jpg')
    gau2 = cv2.cvtColor(gau2, cv2.COLOR_BGR2GRAY)
    sobely_result = convolution(gaussian_blur(gau2, 3), sobFiltery)
    cv2.imwrite('./data/sobely.jpg', sobely_result)
    sobely_result = cv2.imread('./data/sobely.jpg')
    cv2.imshow('after diy sobely', sobely_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def button_event34():
    gau2 = cv2.imread(q3_folder + 'House.jpg')
    gau2 = cv2.cvtColor(gau2, cv2.COLOR_BGR2GRAY)
    sobelx_result = convolution(gaussian_blur(gau2, 3), sobFilterx)
    sobelx_result2 = sobelx_result * (255.0 / sobelx_result.max())
    sobely_result = convolution(gaussian_blur(gau2, 3), sobFiltery)
    sobely_result2 = sobely_result * (255.0 / sobely_result.max())
    sobel_xy = np.sqrt(np.square(sobelx_result2) + np.square(sobely_result2))
    sobel_xy *= 255.0/sobel_xy.max()
    cv2.imwrite('./data/sobelxy.jpg', sobel_xy)
    sobel_xy = cv2.imread('./data/sobelxy.jpg')
    cv2.imshow('after diy sobel_xy', sobel_xy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def button_event41():
    global resize1
    cv2.imshow('resize 256 * 256', resize1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def button_event42():
    global resize1
    # translation matrix
    trans = np.float32([[1, 0, 0], [0, 1, 60]])
    # resize1 = np.array(resize1).astype(np.float32)
    trans1 = cv2.warpAffine(resize1, trans, (400, 300))
    cv2.imshow('after translation', trans1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def button_event43():
    global resize1
    trans = np.float32([[1, 0, 0], [0, 1, 60]])
    # resize1 = np.array(resize1).astype(np.float32)
    resize2 = cv2.warpAffine(resize1, trans, (400, 300))
    # rotational matrix
    h, w = resize2.shape[:2]
    rotate = cv2.getRotationMatrix2D(
        (h / 2, w / 2), 10, 0.5)
    rotate1 = cv2.warpAffine(resize2, rotate, (400, 300))
    cv2.imshow('after rotation', rotate1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def button_event44():
    global resize1
    trans = np.float32([[1, 0, 0], [0, 1, 60]])
    resize2 = cv2.warpAffine(resize1, trans, (400, 300))
    # rotational matrix
    h, w = resize2.shape[:2]
    rotate = cv2.getRotationMatrix2D((h / 2, w / 2), 10, 0.5)
    rotate1 = cv2.warpAffine(resize2, rotate, (400, 300))
    non_shear = np.float32([[50, 50], [200, 50], [50, 200]])
    after_shear = np.float32([[10, 100], [200, 50], [100, 250]])
    shearing_matrix = cv2.getAffineTransform(non_shear, after_shear)
    shear_img = cv2.warpAffine(rotate1, shearing_matrix, (400, 300))
    cv2.imshow('after shearing', shear_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


mybutton1 = tk.Button(root, text='(1-1)Load Image File', command=button_event1)
mybutton2 = tk.Button(root, text='(1-2)Color Separation', command=button_event2)
mybutton3 = tk.Button(root, text='(1-3)Color Transformation', command=button_event3)
mybutton4 = tk.Button(root, text='(1-4)Blending', command=button_event4)
mybutton21 = tk.Button(root, text='(2-1)Gaussian blur ', command=button_event21)
mybutton22 = tk.Button(root, text='(2-2)Bilateral filter  ', command=button_event22)
mybutton23 = tk.Button(root, text='(2-3)Median filter  ', command=button_event23)
mybutton31 = tk.Button(root, text='(3-1)Gaussian Blur', command=button_event31)
mybutton32 = tk.Button(root, text='(3-2)Sobel X ', command=button_event32)
mybutton33 = tk.Button(root, text='(3-3)Sobel Y ', command=button_event33)
mybutton34 = tk.Button(root, text='(3-4)Magnitude ', command=button_event34)
mybutton41 = tk.Button(root, text='(4-1)Resize ', command=button_event41)
mybutton42 = tk.Button(root, text='(4-2)Translation ', command=button_event42)
mybutton43 = tk.Button(root, text='(4-3)Rotation, Scaling ', command=button_event43)
mybutton44 = tk.Button(root, text='(4-4)Shearing ', command=button_event44)


mybutton1.pack()
mybutton1.place(x=20, y=70)
mybutton2.pack()
mybutton2.place(x=20, y=100)
mybutton3.pack()
mybutton3.place(x=20, y=130)
mybutton4.pack()
mybutton4.place(x=20, y=160)


mybutton21.pack()
mybutton21.place(x=200, y=70)
mybutton22.pack()
mybutton22.place(x=200, y=100)
mybutton23.pack()
mybutton23.place(x=200, y=130)


mybutton31.pack()
mybutton31.place(x=380, y=70)
mybutton32.pack()
mybutton32.place(x=380, y=100)
mybutton33.pack()
mybutton33.place(x=380, y=130)
mybutton34.pack()
mybutton34.place(x=380, y=160)


mybutton41.pack()
mybutton41.place(x=560, y=70)
mybutton42.pack()
mybutton42.place(x=560, y=100)
mybutton43.pack()
mybutton43.place(x=560, y=130)
mybutton44.pack()
mybutton44.place(x=560, y=160)


if __name__ == "__main__":
    # for Q4
    original_img = cv2.imread(q4_folder + 'SQUARE-01.png')
    h, w = original_img.shape[: 2]
    resize1 = cv2.resize(original_img, (int(h / 2), int(w / 2)))

    root.mainloop()
