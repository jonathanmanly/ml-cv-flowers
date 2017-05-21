import cv2
import numpy as np
from scipy.spatial import distance
import os
import pandas as pd


hist_depth = 8
stride=60#30
num_templates =18#11

kernel = np.ones((5,5),np.uint8)


templates = []
for x in range(1,num_templates+1):
    filename = "patch/f"+str(x)+".png"
    templates.append(cv2.imread(filename))


t=templates[12]

blurred = cv2.GaussianBlur(t,(5,5),2)
blurred=blurred
grey = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
grey = cv2.Sobel(grey,cv2.CV_64F,1,1,ksize=7)
cv2.imshow("sobel",grey)
edges = cv2.Canny(grey,230,70)
cv2.imshow("edges",edges)
#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#cv2.imshow("close",edges)
edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
cv2.imshow("open",edges)
(cnts,_) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
cv2.drawContours(t, [cnts[0]], -1, (0, 255, 0), 3)


humo = cv2.HuMoments(cv2.moments(edges)).flatten()

print humo

cv2.imshow("edges",edges)
cv2.imshow("blurred",blurred)
cv2.imshow("origwcont",t)
cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.waitKey(0)