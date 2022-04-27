import cv2
import numpy as np
import time
# 4 cc i.e a 4 byte code used for video codec; for windows is XVID; IOS = AVC1

FourCC = cv2.VideoWriter_fourcc(*'XVID')

file = cv2.VideoWriter('file.avi', FourCC, 30, (640, 480))

# starting the camera
camera = cv2.VideoCapture(0)
time.sleep(3)

# for bg images
for i in range(60):
    label, img_data = camera.read()

img_data = np.flip(img_data, axis=1)

while camera.isOpened():
    label, img = camera.read()

    if not label:
        break

    img = np.flip(img, axis=1)

    # converting the color from rgb ( bgr ) to hsl ( hsv )
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_range = np.array([0, 120, 50])
    upper_range = np.array([10, 255, 255])

    lower_mask = cv2.inRange(hsv, lower_range, upper_range)

    lower_range = np.array([170, 120, 50])
    upper_range = np.array([255, 255, 255])

    upper_mask = cv2.inRange(hsv, lower_range, upper_range)

    lower_mask = lower_mask + upper_mask

    # open and expand ( dilation operation) for combined mask with lower shade and upper shade
    lower_mask = cv2.morphologyEx(
        lower_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    lower_mask = cv2.morphologyEx(
        lower_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # selecting the maskless parts of lower_mask and saving in upper_mask

    upper_mask = cv2.bitwise_not(lower_mask)

    # keeping the part of img without red color
    filtered_img = cv2.bitwise_and(img, img, mask=upper_mask)
    filtered_bg = cv2.bitwise_and(img_data, img_data, mask=lower_mask)

    output = cv2.addWeighted(filtered_img, 1, filtered_bg, 1, 0)

    file.write(output)

    cv2.imshow('Haarrryyyy', output)
    cv2.waitKey(2)

camera.release()
cv2.destroyAllWindows()
