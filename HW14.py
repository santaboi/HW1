import cv2
from cfg import *
import numpy as np

original_img = cv2.imread(q4_folder + 'SQUARE-01.png')

h, w = original_img.shape[: 2]
resize1 = cv2.resize(original_img, (int(h / 2), int(w / 2)))
# print(resize1.shape)
cv2.imshow('resize 256 * 256', resize1)
cv2.waitKey(0)
'''
resize2 = cv2.resize(resize1, (2 * h, 2 * w))
cv2.imshow('resize 1028 * 1028', resize2)
cv2.waitKey(0)
'''


# **************************************************************************
resize1 = cv2.resize(original_img, (int(h / 2), int(w / 2)))


# translation matrix
trans = np.float32([[1, 0, 0], [0, 1, 60]])
#resize1 = np.array(resize1).astype(np.float32)
trans1 = cv2.warpAffine(resize1, trans, (400, 300))
cv2.imshow('after translation', trans1)
cv2.waitKey(0)

# **************************************************************************
original_img = cv2.imread(q4_folder + 'SQUARE-01.png')

h, w = original_img.shape[: 2]
resize1 = cv2.resize(original_img, (int(h / 2), int(w / 2)))

# rotational matrix
rotate = cv2.getRotationMatrix2D((h / 4, w / 2), 10, 0.5)  # 中心點是啥???????????
rotate1 = cv2.warpAffine(resize1, rotate, (400, 300))
cv2.imshow('after rotation', rotate1)
cv2.waitKey(0)


# **************************************************************************
original_img = cv2.imread(q4_folder + 'SQUARE-01.png')

'''
h, w = original_img.shape[: 2]
resize1 = cv2.resize(original_img, (int(h / 4), int(w / 4)))
cv2.imshow('after shearing', shear_img)
cv2.waitKey(0)
'''
h, w = original_img.shape[: 2]
resize1 = cv2.resize(original_img, (int(h / 2), int(w / 2)))

# rotational matrix
rotate = cv2.getRotationMatrix2D((h / 4, w / 2), 10, 0.5)  # 中心點是啥???????????
rotate1 = cv2.warpAffine(resize1, rotate, (400, 300))
non_shear = np.float32([[50, 50], [200, 50], [50, 200]])
after_shear = np.float32([[10, 100], [200, 50], [100, 250]])
shearing_matrix = cv2.getAffineTransform(non_shear, after_shear)
shear_img = cv2.warpAffine(rotate1, shearing_matrix, (400, 300))
cv2.imshow('after shearing', shear_img)
cv2.waitKey(0)


# 4-3 4-4 題 參數可能有點怪
