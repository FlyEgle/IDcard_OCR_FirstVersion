# -*- coding:utf-8 -*-
"""
身份证号识别
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import time
from PIL import Image


class OcrOfIdCard(object):
    """
    身份证号识别的类
    """
    def __init__(self):
        self.image = None
        self.threshold = None
        self.grayimage = None
        self.contours = None
        self.key_cont = None
        self.file_name = 'id_num.png'
        self.id_number = None

    def binary(self, original_image, threshold=100):
        """
        :param original_image: 起始图片
        :param threshold: 二值化阈值
        :return: 二值图像,0,1翻转
        """
        self.image = cv2.imread(original_image)
        self.grayimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        w, h = self.grayimage.shape
        for i in range(w):
            for j in range(h):
                if self.grayimage[i, j] <= threshold:
                    self.grayimage[i, j] = 1
                else:
                    self.grayimage[i, j] = 0
        return self.grayimage

    def image_dilate(self, kernel_size):
        """
        :param kernel_size: 滤波器算子大小
        :return: 膨胀后的图像
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.grayimage = cv2.dilate(self.grayimage, kernel=kernel)
        return self.grayimage

    def extract_region(self):
        """
        :return: 身份证号的ROI
        """
        contours_image, contours, hierarchy = cv2.findContours(self.grayimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ROI = cv2.drawContours(self.image, contours, -1, (0, 255, 0), 1)
        self.contours = contours[3]
        return ROI

    def getpoint_ROI(self):
        """
        :return: ROI的最小矩形坐标点
        """
        w, n, h = self.contours.shape
        self.key_cont = np.reshape(self.contours, (w, h))

        xPoints = []
        yPoints = []

        for i in range(w):
            xPoints.append(self.key_cont[i][0])
            yPoints.append(self.key_cont[i][1])

        x_min = np.min(xPoints)
        y_min = np.min(yPoints)
        x_max = np.max(xPoints)
        y_max = np.max(yPoints)

        return x_min, y_min, x_max, y_max

    def draw_rectangle(self):
        """
        :return: 返回矩形ROI
        """
        x1, y1, x2, y2 = self.getpoint_ROI()
        ROI_image = cv2.rectangle(self.image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
        return ROI_image

    def crop_roi(self):
        """
        :return: 身份证号区域
        """
        x1, y1, x2, y2 = self.getpoint_ROI()
        image_crop = self.image[y1+4:y2-4, x1+4:x2-4]
        return image_crop

    def denoise(self):
        """
        降噪和去模糊处理，待定
        :return:
        """
        pass

    def save_png(self):
        """
        保存图片，以后需要详细定义路径和图片像素
        :return:
        """
        image_crop = self.crop_roi()
        cv2.imwrite(filename=self.file_name, img=image_crop)
        print ('save png as id_num.png')

    def ocr_recogniation(self):
        """
        这里用的是pytesseract，后续可以自己用keras训练一个模型
        :return: 身份证号码
        """
        self.save_png()
        self.id_number = pytesseract.image_to_string(Image.open(self.file_name))
        return self.id_number


if __name__ == '__main__':

    time_start = time.time()
    idcard = OcrOfIdCard()
    thresh_image = idcard.binary('1.jpeg', threshold=100)
    dilate_image = idcard.image_dilate(14)
    ROI = idcard.extract_region()
    ROI_image = idcard.draw_rectangle()
    crop_image = idcard.crop_roi()
    code = idcard.ocr_recogniation()
    time_end = time.time()
    print ('The total time is :{}'.format(time_end - time_start), 's')
    print (code)

    plt.subplot(221), plt.imshow(thresh_image, cmap='gray')
    plt.subplot(222), plt.imshow(dilate_image)
    plt.subplot(223), plt.imshow(ROI_image)
    plt.subplot(224), plt.imshow(crop_image)
    plt.show()


