import cv2
import imutils
import numpy as np
import math

image = cv2.imread('D:/imag/P.jpg')
log_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        log_img[i, j, 0] = math.log(1 + image[i, j, 0])
        log_img[i, j, 1] = math.log(1 + image[i, j, 1])
        log_img[i, j, 2] = math.log(1 + image[i, j, 2])
cv2.normalize(log_img, log_img, 0, 255, cv2.NORM_MINMAX)
log_img = cv2.convertScaleAbs(log_img)
cv2.imshow('image', imutils.resize(image, 400))
cv2.imshow('log transform', imutils.resize(log_img, 400))
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()