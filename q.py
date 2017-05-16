import cv2
import numpy as np
from scipy.spatial import distance
import os
import pandas as pd


hist_depth = 8
stride=60  # X/Y size of the segments used to examine the input images
num_templates =18 # Number of templates in directory 'patch'


#Load the prepared patch templates for input, named f1.png - fx.png

templates = []
for x in range(1,num_templates+1):
    filename = "patch/f"+str(x)+".png"
    templates.append(cv2.imread(filename))


# Make a version of each patch template for template matching.  This involves a Gaussian blur
# and transformation to greyscale

greyBlurredTemplates=[]


for t in templates:
    blurred = cv2.GaussianBlur(t,(5,5),1)
    grey = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
    greyBlurredTemplates.append(grey)


# Calculate the Hu moments for each template.  This takes the smoothed greyscale template from above,
# runs Canny edge detection, then calculates the Hu moments (array of len 7).

humoments = []

for t in greyBlurredTemplates:
    edges = cv2.Canny(t,100,200)
    humo = cv2.HuMoments(cv2.moments(edges)).flatten()
    humo = humo/humo.max()
    humoments.append(humo)




def getTemplateHists(img):
    # Convenience function to be used below to create the BGR and HSV histograms for an input segment
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist_img = cv2.calcHist([img], [0,1,2], None, [hist_depth,hist_depth,hist_depth], [120, 255,120,255,120,255])
    hist_hsv = cv2.calcHist([img], [0,1], None, [hist_depth,hist_depth], [0,180,0,255])
    hist_img = cv2.normalize(hist_img,hist_img,0,1,cv2.NORM_MINMAX)
    hist_hsv = cv2.normalize(hist_hsv,hist_hsv,0,1,cv2.NORM_MINMAX)
    return hist_img,hist_hsv





#Number of input images in each directory
dataSources = {'train':2296,'test':1532}

for d in ['train','test']:
    alldata = []
    search_ind_max = dataSources[d]
    search_range = range(1,search_ind_max)
    if d=='train':
        search_range = [x for x in search_range if x not in [3,90,167,226,833,1644,1707,1561]] #remove the pictures that templates came from
    for search_ind in search_range:
        if search_ind<10 or search_ind%100==0:
            #Verbose status update
            print d,search_ind
        #Load the current image, and define the number of segments, based on the stride parameter
        thisData =[search_ind]
        filename = d+"/"+str(search_ind)+".jpg"
        search = cv2.imread(filename)
        U = search.shape[0]/stride
        V= search.shape[1]/stride
        # Create an empty array to store segment-wise statistics in
        results=np.zeros(((stride)**2,8))
        # Create greyscale and HSV versions of the current search image
        hsv_search = cv2.cvtColor(search,cv2.COLOR_BGR2HSV)
        search_grey = cv2.cvtColor(cv2.GaussianBlur(search,(5,5),1),cv2.COLOR_BGR2GRAY)
        for tnum in range(len(templates)):
            t=templates[tnum]
            grey = greyBlurredTemplates[tnum]
            hu_template = humoments[tnum]
            # Find the min (best) and average normed sqare difference between the template and the images
            matching = cv2.matchTemplate(search_grey,grey,cv2.TM_SQDIFF_NORMED)
            thisData.append(matching.min())
            thisData.append(matching.mean())
            thisData.append(len(np.where(matching[:,1]<.3)[0]))
            hist_img,hist_hsv = getTemplateHists(t)
            # Nested double loop uses Numpy indexing to divide the image into stride x stride sized segments
            for y in range(stride):
                for x in range(stride):
                    thisSlice = search[ y*U:(y+1)*U , x*V:(x+1)*V, :]
                    huSlice = search_grey[ y*U:(y+1)*U , x*V:(x+1)*V]
                    sliceHuMo = cv2.HuMoments(cv2.moments(cv2.Canny(huSlice,100,200))).flatten()
                    if sliceHuMo.sum()>0:
                        sliceHuMo= sliceHuMo/sliceHuMo.max()
                    #Calculate the Hu Moments for the segement, and calculate distance from the template Hu Moment
                    huDist = distance.euclidean(sliceHuMo,hu_template)
                    results[y*stride+x,3]=huDist
                    if tnum==1:
                        # Custom blue detector - runs only once per image
                        bigBlue = np.zeros((thisSlice.shape[0],thisSlice.shape[1]))
                        bigBlue[thisSlice[:,:,0]>200]=1
                        bigBlue[thisSlice[:,:,1]>150]=0
                        bigBlue[thisSlice[:,:,2]>150]=0
                        results[y*stride+x,6]=bigBlue.sum()/bigBlue.size
                    #Calculate the segment BGR and HSV histograms.  Norm them and calculate distance to the template for each segment
                    this_hsv_Slice = hsv_search[ y*U:(y+1)*U , x*V:(x+1)*V, :]
                    hist_slice = cv2.calcHist([thisSlice], [0,1,2], None, [hist_depth,hist_depth,hist_depth], [120, 255,120, 255,120, 255])
                    hist_hsv_slice = cv2.calcHist([this_hsv_Slice], [0,1], None, [hist_depth,hist_depth], [0,180,0,255])
                    hist_slice=cv2.normalize(hist_slice,hist_slice,0,1,cv2.NORM_MINMAX)
                    hist_hsv_slice=cv2.normalize(hist_hsv_slice,hist_hsv_slice,0,1,cv2.NORM_MINMAX)
                    dist1= distance.euclidean(np.ravel(hist_slice),np.ravel(hist_img))
                    dist2= distance.euclidean(np.ravel(hist_hsv_slice),np.ravel(hist_hsv))
                    results[y*stride+x,0]=dist1
                    results[y*stride+x,1]=dist2
            #This creates the actual record for the ML algorithm.  It looks across all segments for each template for
            # min and threshold count statistics to store.
            thisData.append(results[:,0].min())
            thisData.append(results[:,1].min())
            thisData.append(len(np.where(results[:,0]<2.)[0]))
            thisData.append(len(np.where(results[:,1]<1.3)[0]))
            thisData.append(len(np.where(results[:,0]<2.5)[0]))
            thisData.append(len(np.where(results[:,1]<1.5)[0]))
            thisData.append(results[:,3].min())
        thisData.append(results[:,6].max())
        alldata.append(thisData)


    # Convert to pandas df, and add variable names, one set per input template
    df = pd.DataFrame(alldata)
    varnames =['img_num']
    for i in range(1,num_templates+1):
        varnames.append("t"+str(i)+"minmatchEucRGB")
        varnames.append("t"+str(i)+"minmatchEucHSV")
        varnames.append("t"+str(i)+"threshnmatch")
        varnames.append("t"+str(i)+"minBGRhist")
        varnames.append("t"+str(i)+"minHSVhist")
        varnames.append("t"+str(i)+"threshBGRhist")
        varnames.append("t"+str(i)+"threshHSVhist")
        varnames.append("t"+str(i)+"threshBGRhist2")
        varnames.append("t"+str(i)+"threshHSVhist2")
        varnames.append("t"+str(i)+"huDist")
    varnames.append("veryBlueProp")
    df.columns = varnames
    df.index = df['img_num']
    df.to_csv(d+"photo_features.csv")










