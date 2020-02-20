#!/usr/bin/env python
# coding: utf-8

import keras
import cv2
import numpy as np
import matplotlib

def draw_edges(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

    canny_edges = cv2.Canny(img_gray_blur, 20, 50)
    _, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)

    return mask


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.imshow('Stream', draw_edges(frame))
    if cv2.waitKey(1) == 13:
        break


cap.release()
cv2.destroyAllWindows()
