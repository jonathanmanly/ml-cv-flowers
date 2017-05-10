import cv2
import numpy as np
from scipy.spatial import distance
import os
import pandas as pd


hist_depth = 8
stride=30
num_templates =11


templates = []
for x in range(1,num_templates+1):
    filename = "patch/f"+str(x)+".png"
    templates.append(cv2.imread(filename))


greyBlurredTemplates=[]


for t in templates:
    blurred = cv2.GaussianBlur(t,(5,5),1)
    grey = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
    greyBlurredTemplates.append(grey)



def getTemplateHists(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist_img = cv2.calcHist([img], [0,1,2], None, [hist_depth,hist_depth,hist_depth], [120, 255,120,255,120,255])
    hist_hsv = cv2.calcHist([img], [0,1], None, [hist_depth,hist_depth], [0,180,0,255])
    hist_img = cv2.normalize(hist_img,hist_img,0,1,cv2.NORM_MINMAX)
    hist_hsv = cv2.normalize(hist_hsv,hist_hsv,0,1,cv2.NORM_MINMAX)
    return hist_img,hist_hsv




alldata = []

dataSources = {'train':2296,'test':1531}#2296 #1531

for d in ['train','test']:
    search_ind_max = dataSources[d]
    for search_ind in range(1,search_ind_max):
        if search_ind<10 or search_ind%100==0:
            print d,search_ind
        thisData =[search_ind]
        filename = d+"/"+str(search_ind)+".jpg"
        search = cv2.imread(filename)
        U = search.shape[0]/stride
        V= search.shape[1]/stride
        results=np.zeros(((stride)**2,3))
        hsv_search = cv2.cvtColor(search,cv2.COLOR_BGR2HSV)
        search_grey = cv2.cvtColor(cv2.GaussianBlur(search,(5,5),1),cv2.COLOR_BGR2GRAY)
        for tnum in range(len(templates)):
            t=templates[tnum]
            grey = greyBlurredTemplates[tnum]
            matching = cv2.matchTemplate(search_grey,grey,cv2.TM_SQDIFF_NORMED)
            thisData.append(matching.min())
            thisData.append(matching.mean())
            thisData.append(len(np.where(matching[:,1]<.3)[0]))
            hist_img,hist_hsv = getTemplateHists(t)
            for y in range(stride):
                for x in range(stride):
                    thisSlice = search[ y*U:(y+1)*U , x*V:(x+1)*V, :]
                    if tnum==1:
                        bigBlue = np.zeros((thisSlice.shape[0],thisSlice.shape[1]))
                        bigBlue[thisSlice[:,:,0]>200]=1
                        bigBlue[thisSlice[:,:,1]>150]=0
                        bigBlue[thisSlice[:,:,2]>150]=0
                        results[y*stride+x,2]=bigBlue.sum()/bigBlue.size
                    this_hsv_Slice = hsv_search[ y*U:(y+1)*U , x*V:(x+1)*V, :]
                    hist_slice = cv2.calcHist([thisSlice], [0,1,2], None, [hist_depth,hist_depth,hist_depth], [120, 255,120, 255,120, 255])
                    hist_hsv_slice = cv2.calcHist([this_hsv_Slice], [0,1], None, [hist_depth,hist_depth], [0,180,0,255])
                    hist_slice=cv2.normalize(hist_slice,hist_slice,0,1,cv2.NORM_MINMAX)
                    hist_hsv_slice=cv2.normalize(hist_hsv_slice,hist_hsv_slice,0,1,cv2.NORM_MINMAX)
                    dist1= distance.euclidean(np.ravel(hist_slice),np.ravel(hist_img))
                    dist2= distance.euclidean(np.ravel(hist_hsv_slice),np.ravel(hist_hsv))
                    results[y*stride+x,0]=dist1
                    results[y*stride+x,1]=dist2
            thisData.append(results[:,0].min())
            thisData.append(results[:,1].min())
            thisData.append(len(np.where(results[:,0]<2.)[0]))
            thisData.append(len(np.where(results[:,1]<1.3)[0]))
            thisData.append(len(np.where(results[:,0]<2.5)[0]))
            thisData.append(len(np.where(results[:,1]<1.5)[0]))
        thisData.append(results[:,2].max())
        alldata.append(thisData)


        df = pd.DataFrame(alldata)
        varnames =['img_num']
        for i in range(1,num_templates+1):
            varnames.append("t"+str(i)+"minmatch")
            varnames.append("t"+str(i)+"meannmatch")
            varnames.append("t"+str(i)+"threshnmatch")
            varnames.append("t"+str(i)+"minBGRhist")
            varnames.append("t"+str(i)+"minHSVhist")
            varnames.append("t"+str(i)+"threshBGRhist")
            varnames.append("t"+str(i)+"threshHSVhist")
            varnames.append("t"+str(i)+"threshBGRhist2")
            varnames.append("t"+str(i)+"threshHSVhist2")

        varnames.append("veryBlueProp")
        df.columns = varnames
        df.index = df['img_num']
        df.to_csv(d+"photo_features.csv")










