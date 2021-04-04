import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
from IPython.display import display, Markdown

# Читання
img = cv2.imread('road.jpg')
cv2.imshow('Зображення з відео', img)
cv2.waitKey()
cv2.destroyAllWindows()

# перетворення зображення в чорно-біле
img1 = cv2.imread('road.jpg', cv2.COLOR_BGR2GRAY)
cv2.imwrite('1_line_gray.jpg', img1)

# розмивання за фільтром Гауса
blur_kernel_size = (5, 5)
gray_blur = cv2.GaussianBlur(img1, blur_kernel_size, 2)
cv2.imwrite('1_line_Blur.jpg', gray_blur)


# Виділення контурів алгоритмом Кенні
canny_image = cv2.Canny(gray_blur, 50, 100)
cv2.imshow('Blur + Canny', canny_image)
cv2.imwrite('1_line_canny.jpg', canny_image)
cv2.waitKey()
cv2.destroyAllWindows()


lines = cv2.HoughLinesP(canny_image, 1, np.pi/180, 90, None, 50, 10)
lines = [] if lines is None else lines
for (x1, y1, x2, y2) in map(lambda x: x[0], lines):
    # 1280 720
    if y1 < 300 or y2 < 300 or y1 > 700 or y2 > 700:
        continue
    if x1 != x2 and abs((y2-y1)/(x2-x1)) < 0.2:
        continue
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5, cv2.LINE_AA)

cv2.imshow("HoughLines", img)
cv2.waitKey()
cv2.destroyAllWindows()

