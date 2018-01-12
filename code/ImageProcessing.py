# -*- coding:utf-8 -*-
"""
this class is for original ID card Image Processing
Normalization, Resize, DeBlur, Transformer
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageGetIdCard(object):

    def __init__(self):
        pass


img = cv2.imread('../image/3.jpg')
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grayimg, 200, 255, cv2.THRESH_BINARY)
_, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

x_point = []
y_point = []
width = []
height = []
square = []
for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    width.append(w)
    height.append(h)
    x_point.append(x)
    y_point.append(y)
    square.append(w*h)

maxrectangle = int(np.argmax(square))
x_max = x_point[maxrectangle]
y_max = y_point[maxrectangle]
w_max = width[maxrectangle]
h_max = height[maxrectangle]
cv2.rectangle(img, (x_max, y_max), (x_max+w_max, y_max+h_max), (0, 255, 0), 1)

cardimage = img[y_max+30:y_max+h_max-30, x_max+30:x_max+w_max-30]
cv2.imshow('cardimage', cardimage)
cv2.imwrite('../image/3_roi.jpg', cardimage)
cv2.waitKey(0)
