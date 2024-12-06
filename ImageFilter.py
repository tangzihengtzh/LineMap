'''
图片处理函数源自网络
'''
import cv2
import numpy as np


# 图像轮廓检测处理
def edge(img):
    gray = cv2.GaussianBlur(img, (5, 5), 0)
    result = cv2.Canny(gray, 100, 200)
    return result


# 图像模糊处理
def blur(img):
    result = cv2.blur(img, (15, 15))
    return result


# 图像锐化处理
def sharp(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    result = cv2.filter2D(img, -1, kernel)
    return result


# 图像双边滤波处理(美颜)
def bifilter(img):
    result = cv2.bilateralFilter(src=img, d=0, sigmaColor=30, sigmaSpace=15)
    return result


# 图像浮雕处理
def relief(img, Degree):  # 参数为原图像和浮雕图像程度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[0:2]
    # 定义空白图像，存放图像浮雕处理之后的图片
    img1 = np.zeros((h, w), dtype=gray.dtype)
    # 通过对原始图像进行遍历，通过浮雕公式修改像素值，然后进行浮雕处理
    for i in range(h):
        for j in range(w - 1):
            # 前一个像素值
            a = gray[i, j]
            # 后一个像素值
            b = gray[i, j + 1]
            # 新的像素值,防止像素溢出
            img1[i, j] = min(max((int(a) - int(b) + Degree), 0), 255)
    return img1


#  图像素描处理
def Sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 通过高斯滤波过滤噪声
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    # 通过canny算法提取图像轮过
    canny = cv2.Canny(gaussian, 50, 140)
    # 对轮廓图像进行反二进制阈值化处理
    ret, result = cv2.threshold(canny, 90, 255, cv2.THRESH_BINARY_INV)
    return result


# 图像怀旧处理
def Nostalgia(img):  # 参数为原图像
    # 获取图像属性
    h, w = img.shape[0:2]
    # 定义空白图像，存放图像怀旧处理之后的图片
    img1 = np.zeros((h, w, 3), dtype=img.dtype)
    # 通过对原始图像进行遍历，通过怀旧公式修改像素值，然后进行怀旧处理
    for i in range(h):
        for j in range(w):
            B = 0.131 * img[i, j, 0] + 0.534 * img[i, j, 1] + 0.272 * img[i, j, 2]
            G = 0.168 * img[i, j, 0] + 0.686 * img[i, j, 1] + 0.349 * img[i, j, 2]
            R = 0.189 * img[i, j, 0] + 0.769 * img[i, j, 1] + 0.393 * img[i, j, 2]
            # 防止图像溢出
            if B > 255:
                B = 255
            if G > 255:
                G = 255
            if R > 255:
                R = 255
            img1[i, j] = [int(B), int(G), int(R)]  # B\G\R三通道都设置为怀旧值
    return img1


# 图像水彩画效果处理
def stylization(img):
    result = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    return result

