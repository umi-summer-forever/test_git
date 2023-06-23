import matplotlib.pyplot as plt
import numpy as np
import cv2


path = 'XXXX/'
img  = cv2.imread(path + 'D:/imag/P.jpg')

img_gray       = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray_array = np.array(img_gray)

zmax = int(img_gray_array.max())
zmin = int(img_gray_array.min())

t    = (zmax + zmin)/2

while True:

    img_zo = np.where(img_gray_array > t, 0, img_gray_array)
    img_bo = np.where(img_gray_array < t, 0, img_gray_array)

    zo = np.sum(img_zo)/np.sum(img_zo != 0)
    bo = np.sum(img_bo)/np.sum(img_bo != 0)

    k = (zo + bo)/2

    if abs(t - k) < 0.01:
        break;
    else:
        t = k


img_gray_array[img_gray_array > t]  = 255
img_gray_array[img_gray_array <= t] = 0

plt.imshow(img_gray_array, cmap='gray')
plt.show()
