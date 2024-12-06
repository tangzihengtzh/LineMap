# import cv2
# import numpy as np
#
#
# def line_add(canvas, pt1, pt2, color, thickness=1):
#     """
#     在画布上绘制直线，通过增加像素值而不是直接覆盖。
#
#     参数:
#         canvas (numpy.ndarray): 画布图像。
#         pt1 (tuple): 直线起点 (x1, y1)。
#         pt2 (tuple): 直线终点 (x2, y2)。
#         color (tuple): 要增加的颜色 (r, g, b)。
#         thickness (int): 线条的厚度。
#
#     返回:
#         numpy.ndarray: 更新后的画布。
#     """
#     # 创建一张与canvas相同大小的全0图像
#     temp_canvas = np.zeros_like(canvas, dtype=np.int32)
#
#     # 在temp_canvas上绘制直线
#     cv2.line(temp_canvas, pt1, pt2, color, thickness)
#
#     # 将temp_canvas加到原canvas上
#     canvas = cv2.add(canvas.astype(np.int32), temp_canvas)
#
#     # 确保像素值不超过255，防止溢出
#     canvas = np.clip(canvas, 0, 255).astype(np.uint8)
#
#     return canvas
#
#
# # 示例使用
# canvas = np.zeros((400, 400, 3), dtype=np.uint8)
# canvas = line_add(canvas, (50, 50), (350, 350), (0, 255, 0), 2)
# cv2.imshow('Canvas', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import matplotlib.pyplot as plt
import ImageFilter as filter
img = cv2.imread('your_image.jpg')
relief = filter.Sketch(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
relief = cv2.cvtColor(relief, cv2.COLOR_BGR2RGB)
plt.imshow(relief)
plt.title('浮雕')
plt.axis('off')
plt.show()