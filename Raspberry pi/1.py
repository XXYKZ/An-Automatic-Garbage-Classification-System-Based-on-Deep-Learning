import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture(0)
time.sleep(1)

while True:
    ret,frame = cap.read()
    cv.imshow('frame',frame)
    if cv.waitKey(1)&0xFF == ord('q'):
        break
    
cv.waitKey(0)
cv.destroyAllWindows()
