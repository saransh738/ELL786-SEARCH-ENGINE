#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imported Libraries
import cv2 
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import time 
from timeit import default_timer as timer


# reading query image
img1 = cv2.imread('downloads/Taj Mahal/Image_3.jpeg') 
# Normalizing the image
img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# Creating sift object
sift = cv2.SIFT_create()
# Finding keypoints and decriptors for the query image
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

i=0
images,keypoint_list, descriptor_list = [],[],[]
for folder in os.listdir("downloads"):
    for image in os.listdir("downloads/"+folder):
        # Reading Images from the dataset
        img = cv2.imread(os.path.join("downloads",folder,image))
        # Normalizing the image
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Finding keypoints and decriptors for image using sift algorithm
        keypoints_2, descriptors_2 = sift.detectAndCompute(img,None)
        keypoint_list.append(keypoints_2)
        descriptor_list.append(descriptors_2)
        images.append(img)
    print(i)
    i+=1


# In[2]:


# match length stores matches and images for the corresponding query image with other images
match_lengths = []
# Reading image folder
i = 0

start = timer()
for folder in os.listdir("downloads"):
    
    for image in os.listdir("downloads/"+folder):
        keypoints_2, descriptors_2 = keypoint_list[i],descriptor_list[i]
        img = images[i]
        index = 1
        i_pr = dict(algorithm = index, trees = 5)
        s_pr = dict(checks = 50)
        # run flann based matcher
        fln = cv2.FlannBasedMatcher(i_pr,s_pr)
        # perform knn match using the descriptors of of query image and the test image
        knn_mth = fln.knnMatch(descriptors_1,descriptors_2,k=2)
        # filter out bad matches via ratio test.
        filter_match = []
        for m,n in knn_mth:
            if m.distance < 0.75*n.distance:
                filter_match.append(m)

        threshold = 100
        # if # of good matches > threshold append to match_lengths
        if len(filter_match)>threshold:
            match_lengths.append([filter_match,img,keypoints_2,descriptors_2,folder])
        i += 1
    

# now we sort the list of matches according to length of matches to find the best matches
match_lengths.sort(key = lambda x: len(x[0]),reverse= True)

# loop over all the good matches
for i in range(len(match_lengths)):
    img = match_lengths[i][1]
    filter_match = match_lengths[i][0]
    keypoints_2, descriptors_2 = match_lengths[i][2],match_lengths[i][3]
    folder = match_lengths[i][4]
    # find all the source and destination points according to the filter_match list
    s_point = np.float32([ keypoints_1[m.queryIdx].pt for m in filter_match ]).reshape(-1,1,2)
    d_point = np.float32([ keypoints_2[m.trainIdx].pt for m in filter_match ]).reshape(-1,1,2)
    # find homograpgy projection by using the source and destination points
    P, Q = cv2.findHomography(s_point, d_point, cv2.RANSAC,5.0)
    # the matches obtained after homography projection
    projection_matches = Q.ravel().tolist()
    # dimensions of the image
    height,weight,depth = img1.shape
    # find the corresponding points on the image
    points = np.float32([ [0,0],[0,height-1],[weight-1,height-1],[weight-1,0] ]).reshape(-1,1,2)
    # use these points to find the perspective transform
    p_tran = cv2.perspectiveTransform(points,P)
    # reconstruct the image using the points and the perspective transform
    img = cv2.polylines(img,[np.int32(p_tran)],True,255,3, cv2.LINE_AA)

    # set the parameters to draw the flann matches
    parameters = dict(matchColor = (0,0,255),singlePointColor = None,matchesMask = projection_matches,flags = 2)
    # draw the image using the parameters showing the matches between the two images
    match_image = cv2.drawMatches(img1,keypoints_1,img,keypoints_2,filter_match,None,**parameters)
    # change image format from BGR to RGB so that we can display using matplotlib
    match_image = cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)
    # display the image matcher
    plt.imshow(match_image, 'gray'),plt.show()
    print(folder)
end = timer()
print("TIME_TAKEN =",end-start)


# In[ ]:





# In[ ]:




