import cv2
import numpy as np
import matplotlib.pyplot as plt
 
img = cv2.imread('D:/imag/P.jpg')
 
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height = grayImage.shape[0]
width = grayImage.shape[1]
 
result = np.zeros((height, width), np.uint8)
 
for i in range(height):
    for j in range(width):
        gray = 255 - grayImage[i,j]
        result[i,j] = np.uint8(gray)
 
 
cv2.imshow("Gray Image", grayImage)
cv2.imshow("Result", result)
 
 
cv2.waitKey(0)
cv2.destroyAllWindows()
 

