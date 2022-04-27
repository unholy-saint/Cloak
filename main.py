import cv2
import numpy as np

size = (640, 480)

Fourcc = cv2.VideoWriter_fourcc(*'XVID')
outputFile = cv2.VideoWriter('file.avi', Fourcc, 30, size)

camera = cv2.VideoCapture(0)

bg = cv2.imread('assets/BG.jpg')
bg = cv2.resize(bg, size)

while camera.isOpened():
    label , img = camera.read()

    if not label:
        break

    img = np.flip(img, axis=1)
    img = cv2.resize(img, size)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lowerMaskRange = np.array([0,0,0])
    upperMaskRange = np.array([130, 14, 30])

    lowerMask = cv2.inRange(img, lowerMaskRange, upperMaskRange)

    # lowerMaskRange = np.array([30,30, 30])
    # upperMaskRange = np.array([104, 153, 70])

    # upperMask = cv2.inRange(img, lowerMaskRange, upperMaskRange)

    # frame = upperMask + lowerMask

    Morph = cv2.morphologyEx(lowerMask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    Morph = cv2.morphologyEx(Morph, cv2.MORPH_DILATE, np.ones((2,2), np.uint8))

    nonMask = cv2.bitwise_not(Morph)

    filteredImage = cv2.bitwise_and(img, img, mask=nonMask)
    filteredBG = cv2.bitwise_and(bg, bg, mask=Morph)

    output = cv2.addWeighted(filteredImage, 1, filteredBG, 1, 0)
    
    outputFile.write(output)

    cv2.imshow('Bangkok', output)
    cv2.waitKey(3)

camera.release()
cv2.destroyAllWindows()
