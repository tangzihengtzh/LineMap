import cv2
import numpy as np
import random
from tqdm import tqdm
import ImageFilter
import concurrent.futures

def line_add(canvas, pt1, pt2, color, thickness=1):

    """
    在画布上绘制直线，通过增加像素值而不是直接覆盖。

    参数:
        canvas (numpy.ndarray): 画布图像。
        pt1 (tuple): 直线起点 (x1, y1)。
        pt2 (tuple): 直线终点 (x2, y2)。
        color (tuple): 要增加的颜色 (r, g, b)。
        thickness (int): 线条的厚度。

    返回:
        numpy.ndarray: 更新后的画布。
    """
    # 创建一张与canvas相同大小的全0图像
    temp_canvas = np.zeros_like(canvas, dtype=np.int32)

    # 在temp_canvas上绘制直线
    cv2.line(temp_canvas, pt1, pt2, color, thickness)

    # 将temp_canvas加到原canvas上
    canvas = cv2.add(canvas.astype(np.int32), temp_canvas)

    # 确保像素值不超过255，防止溢出
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    return canvas

# 单独处理每个点的函数
def process_point(point, gray_image, length):
    y, x = point
    # 计算x方向和y方向的梯度
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算切线方向：梯度方向旋转90度
    gradient_direction = np.arctan2(grad_y[y, x], grad_x[y, x])
    tangent_direction = gradient_direction + np.pi / 2

    # 延长直线到画布边缘
    x1 = int(x + length * np.cos(tangent_direction))
    y1 = int(y + length * np.sin(tangent_direction))
    x2 = int(x - length * np.cos(tangent_direction))
    y2 = int(y - length * np.sin(tangent_direction))

    return (x1, y1, x2, y2)

# 读取图片并转换为灰度图
image = cv2.imread('test2.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray_image, (21, 21), 0)

# 边缘检测（Canny算法）
edges = cv2.Canny(gray_image, 50, 150)
# edges = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)

# inverted_image = cv2.bitwise_not(gray_image)
# blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)
# inverted_blurred = cv2.bitwise_not(blurred_image)
# edges = cv2.divide(gray_image, inverted_blurred, scale=256.0)

# edges = ImageFilter.Sketch(image)
# edges = cv2.Canny(edges, 50, 150)

# ======================================
# edges = cv2.GaussianBlur(edges, (11, 11), 0)

# 找到所有的边缘点
edge_points = np.column_stack(np.where(edges > 0))

# 随机选取N个边缘点
N = 500  # 可根据需要调整
selected_points = edge_points[random.sample(range(len(edge_points)), N)]

# 创建一个空白画布，大小与原图相同
canvas = np.zeros_like(image)

# # 计算每个点的直线
# length = max(canvas.shape[0], canvas.shape[1])
# # 使用多线程处理
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     results = list(tqdm(executor.map(lambda p: process_point(p, gray_image, length), selected_points), desc="Processing points", total=len(selected_points)))
# # 将所有结果绘制到画布上
# for x1, y1, x2, y2 in tqdm(results,desc="Processing"):
#     canvas = line_add(canvas, (x1, y1), (x2, y2), (5, 5, 5), 1)
# 计算并绘制每个选取点的切线
for point in tqdm(selected_points, desc="Processing points"):
    y, x = point
    # 计算x方向和y方向的梯度
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算切线方向：梯度方向旋转90度
    gradient_direction = np.arctan2(grad_y[y, x], grad_x[y, x])
    tangent_direction = gradient_direction + np.pi / 2

    # 计算切线方向的角度
    angle = np.degrees(tangent_direction)

    # 延长直线到画布边缘
    length = max(canvas.shape[0], canvas.shape[1])
    x1 = int(x + length * np.cos(tangent_direction))
    y1 = int(y + length * np.sin(tangent_direction))
    x2 = int(x - length * np.cos(tangent_direction))
    y2 = int(y - length * np.sin(tangent_direction))

    # 在画布上绘制直线
    # cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    canvas = line_add(canvas, (x1, y1), (x2, y2), (100, 100, 100), 1)

# 显示结果
# cv2.imshow('Original Image', image)
# cv2.imshow('Edges', edges)
# canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
cv2.imshow('Tangents on Canvas', canvas)

# 按任意键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('tangents_on_canvas.jpg', canvas)
