import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
'''/Users/juntysun/Downloads/cup.mp4'''
cap = cv.VideoCapture('F:\ProjectTrash\cup.mp4')

ret, frame = cap.read()
#r,h,c,w = 250,90,400,125
r,h,c,w = 122,265,305,110
track_window = (c,r,w,h)
roi = frame[r:r+h, c:c+w]

hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((25.,25.,0.)), np.array((180.,180.,180.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0,180])



cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        cv.imshow('hsv', hsv)
        cv.imshow('dst', dst)
        #cv.waitKey(0)

        ret, track_window = cv.CamShift(dst, track_window, term_crit)



        pts = cv.boxPoints(ret)

        pts = np.int0(pts)


        img2 = cv.polylines(frame,[pts],True, 255,2)

        cv.imshow('img2',img2)

        '''
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv.imshow('img2', img2) 
        '''

        k = cv.waitKey(30) & 0xff
        #print(k)
        if k == 27:
            break
    else:
        break

cv.destroyAllWindows()
cap.release()
