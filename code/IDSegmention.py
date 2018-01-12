# -*- coding:utf-8 -*-
"""
This Class is the proprecessing for IDcard Segmention
"""
import cv2
import numpy as np


class IdCardExtract(object):
    """
    ksize  and threshold need change find a fit for all same size image
    :returns
        image_proprecessing function is include RGB->GRAY->THRESHOLD, Blur, Transformer, return a image
        imageSegmention function is return a list of Segment
        getRoi function return a list of ROI image
    """

    def __init__(self):
        self.original_image = None
        self.image_path = '../image/3_roi.jpg'
        self.ksize = 20
        self.element = np.ones((self.ksize, self.ksize), np.uint8)
        self.x_point = []
        self.y_point = []
        self.width = []
        self.height = []
        self.ratio = []

    def image_handle(self):
        """
        original->gray->threshold->dilate
        :return: dilate image
        """
        self.original_image = cv2.imread(self.image_path)
        # resize_image = cv2.resize(self.original_image, (640, 400), cv2.INTER_LINEAR)
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        ret, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        dilate_image = cv2.dilate(thresh_image, kernel=self.element)
        return dilate_image

    def image_seg(self):
        """
        find contours and return a contours list
        :return: lists of points and width, height
        """
        image = self.image_handle()
        _, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # self.original_image = cv2.drawContours(self.original_image, contours, -1, (0, 255, 0), 2)
        # the biggest rectangle of Contours
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            self.x_point.append(x)
            self.y_point.append(y)
            self.width.append(w)
            self.height.append(h)
            self.ratio.append(np.float(np.float(w)/np.float(h)))
            self.original_image = cv2.rectangle(self.original_image, (x, y), (x+w, y+h), (0, 0, 0), 1)

        return self.x_point, self.y_point, self.width, self.height

    def get_number_roi(self):
        """
        the max ( width / height ) rectangle is IDcard number
        :return: the image of IDcard Number
        """
        # use image_seg function to catch x, y, w, h
        x_point, y_point, width, height = self.image_seg()
        maxRatioIndex = np.argmax(self.ratio)

        x = self.x_point[int(maxRatioIndex)]
        y = self.y_point[int(maxRatioIndex)]
        h = self.height[int(maxRatioIndex)]
        w = self.width[int(maxRatioIndex)]

        image = cv2.imread(self.image_path)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        id_image = image[y+5:y+h-5, x+3:x+w-3]

        return image, id_image


if __name__ == '__main__':
    idcard_demo = IdCardExtract()
    drawimage, id_image = idcard_demo.get_number_roi()
    cv2.imwrite('../image/id.png', id_image)
    # cv2.imshow('image', drawimage)
    # cv2.imshow('id', id_image)
    # cv2.waitKey(0)
