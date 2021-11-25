import matplotlib.pyplot as plt
import cv2
import numpy as np
#import PyQt5
from cfg import *

# hw1-1
img1_path = q1_folder + 'Sun.jpg'
img1 = cv2.imread(img1_path)
cv2.imshow('Sun.jpg', img1)
print('Sun_width : ', img1.shape[1])
print('Sun_height :', img1.shape[0])
cv2.waitKey(0)
# close window
cv2.destroyWindow('Sun.jpg')

# hw1-2
blue, green, red = cv2.split(img1)
zeros = np.zeros(blue.shape[:2], dtype='uint8')
cv2.imshow('B channel', cv2.merge([blue, zeros, zeros]))
cv2.imshow('G channel', cv2.merge([zeros, green, zeros]))
cv2.imshow('R channel', cv2.merge([zeros, zeros, red]))
cv2.waitKey(0)


# hw 1-3
merge1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
print(merge1.shape)
cv2.imshow('merge1', merge1)
cv2.imshow('merge!!!!', cv2.merge([merge1, merge1, merge1]))


# kind of weird


new_s = img1.copy()
row_s, col_s = img1.shape[0:2]
for i in range(row_s):
    for j in range(col_s):
        new_s[i, j] = sum(img1[i, j]) / 3
"""merge2 = np.zeros((img1.shape[0], img1.shape[1]))
for w in range(0, img1.shape[0]):
    for h in range(0, img1.shape[1]):
        merge2[w, h] = sum(img1[w, h]) / 3"""

# print(merge2)

# way2
cv2.imshow('!!!', new_s)
cv2.waitKey(0)
"""cv2.imwrite('./data/merge2.jpg', cv2.merge([blue / 3, green / 3, red / 3]))
cv2.imshow('./data/merge2',  cv2.merge([blue / 3, green / 3, red / 3]))
cv2.waitKey(0)"""


# ***hw 1-4 (blend and trackbar)***
#dst = src1 * alpha + src2 * beta + gamma
alpha1 = 0.5
src1 = cv2.imread(q1_folder + 'Dog_Strong.jpg')
src2 = cv2.imread(q1_folder + 'Dog_Weak.jpg')
blend3 = cv2.addWeighted(src1, alpha1, src2, (1-alpha1), 0)


def trackbar(a):
    print(a)
    global alpha1, src1, src2, blend3
    alpha1 = cv2.getTrackbarPos('Blend', 'dogs blend(press q to exit)')
    alpha1 = alpha1 * (1/255)
    blend3 = cv2.addWeighted(src1, alpha1, src2, (1-alpha1), 0)
    #blend3 = np.uint8(np.clip(src1 * alpha1 + src2 * (1 - alpha1) + 0), 0, 255)


cv2.namedWindow('dogs blend(press q to exit)')
cv2.createTrackbar('Blend', 'dogs blend(press q to exit)', 0, 255,  trackbar)
cv2.setTrackbarPos('Blend', 'dogs blend(press q to exit)', 255)
while(True):
    cv2.imshow('dogs blend(press q to exit)', blend3)
    if(cv2.waitKey(1) == ord('q')):  # (press q to exit)
        break
cv2.waitKey(0)
