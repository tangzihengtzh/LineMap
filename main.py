import cv2
import numpy as np

# 读取图片
image = cv2.imread('your_image.jpg')

# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯滤波减少噪声
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 使用Canny边缘检测
edges = cv2.Canny(blurred_image, 50, 150)
edges = cv2.GaussianBlur(edges, (3, 3), 0)
# 显示原图和边缘图
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)

# 按任意键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存边缘检测结果
cv2.imwrite('edges_output.jpg', edges)

import cv2
import numpy as np

# 读取图片
image = edges

# 计算x方向和y方向的梯度
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度的方向（以弧度为单位）
gradient_direction = np.arctan2(grad_y, grad_x)

# 计算切线方向：旋转90度
tangent_direction = gradient_direction + np.pi / 2

# 转换到度数（以便更容易理解）
tangent_direction_deg = np.degrees(tangent_direction)

# 将结果限制在0到180度范围内
tangent_direction_deg = np.mod(tangent_direction_deg, 180)

# 输出结果
print("Tangent direction (in degrees):")
print(tangent_direction_deg)

# 显示原图和梯度方向
cv2.imshow('Original Image', image)
cv2.imshow('Tangent Direction', tangent_direction_deg / 180)  # 显示为0到1的范围

# 按任意键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
