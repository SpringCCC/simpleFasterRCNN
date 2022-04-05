# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 10:45
# @Author  : WeiHuang

import cv2
import numpy as np

img_path = r"D:\small.jpg"
img = cv2.imread(img_path)
img = img[:, :, ::-1].copy()
# img /= 255.0
cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), 2)
cv2.putText(img, "huangwei", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)


cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.waitKey(0)

# cv2.destroyAllWindows()


# img = cv2.imread(img_path)
# print(img.shape)
# cv2.rectangle(img, (240, 0), (480, 375), (0, 0, 255), 2)
# cv2.imshow("fff", img)
