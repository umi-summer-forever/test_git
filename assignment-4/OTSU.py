import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('D:/imag/P.jpg')

hist = cv.calcHist([img],[0],None,[256],[0,256])

plt.plot(hist)
plt.show()
