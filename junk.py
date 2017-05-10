
#from matplotlib import pyplot as plt

'''

img=cv2.imread("patch/f1.png")
neg=cv2.imread("train/1.jpg")
pos=cv2.imread("train/3.jpg")

hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hist_img = cv2.calcHist([hsv_img], [0], None, [5], [0, 180])
print hist_img
hist_img=hist_img/hist_img.sum()

hsv_neg = cv2.cvtColor(neg,cv2.COLOR_BGR2HSV)
hist_neg = cv2.calcHist([hsv_neg], [0], None, [5], [0, 180])
print hist_neg
hist_neg=hist_neg/hist_neg.sum()

hsv_pos = cv2.cvtColor(pos,cv2.COLOR_BGR2HSV)
hist_pos = cv2.calcHist([hsv_pos], [0], None, [5], [0, 180])
print hist_pos
hist_pos=hist_pos/hist_pos.sum()

print hist_img
print hist_neg
print hist_pos


a=distance.euclidean(hist_img,hist_neg)
b=distance.euclidean(hist_img,hist_pos)


'''



