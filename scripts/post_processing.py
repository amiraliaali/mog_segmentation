import cv2
import numpy as np

def post_processing(binary_mask):
    binary_mask = cv2.imread("segmented_mask.png", cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((3, 3), np.uint8)

    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=3)

    return dilated_mask
