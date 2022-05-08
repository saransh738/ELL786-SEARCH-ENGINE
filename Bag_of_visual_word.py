#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import all the necessary libraries
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import time
from timeit import default_timer as timer
import cv2
import os

#store images in image list
#store descriptors in Desc_2D => made by extend function
image_list,tag_list ,Desc_2D,Descri_3D= [],[],[],[]
#store descriptors in Desc_3D => made by append function


j=0  
# creating feature_sift using sift library in cv2 to extract features.
feature_sift = cv2.SIFT_create() 
#importing all images from folder downloads
for folder_name in os.listdir("downloads"):
    for image in os.listdir("downloads/"+folder_name):
        #reading respective images
        img = cv2.imread(os.path.join("downloads",folder_name,image))
        #normalizing the given image
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #appending image to image_list
        image_list.append(img)
        tag_list.append(folder_name)
        #now compute the keypoint and descriptors
        kept, descr = feature_sift.detectAndCompute(img,None) 
        #appending and extending descriptors
        Desc_2D.extend(descr)
        Descri_3D.append(descr)
        
    #incrementing j   
    j+=1 
    
# number of images    
N_images= len(image_list) 
#number of centres in kmeans
N_cluster=50
print("input images have been stored in a list")


# In[2]:


#applying kmeans algorithm to find k clusters in descriptor list desc_2D 
K_Mean = KMeans(n_clusters=N_cluster,n_init=2,max_iter=100, verbose=True).fit(Desc_2D)
print("K means ended")  #kmean algorithm ended


# In[3]:


# initiating histogram for all descriptors using k-centres
Histogram = np.array([np.zeros(N_cluster) for i in range(N_images)])
#initiating i
i=0 
while (i<N_images):
    j=0
    while (j <len(Descri_3D[i])):
        # for i, jth element in descripto list 3d
        Desc_1D = Descri_3D[i][j]   
        j+=1
        #predicting the respected centroid
        pr = K_Mean.predict([Desc_1D])
        #incrementing the count of respective centroid
        Histogram[i][pr] += 1
    i+=1   #incrementing i
       
print("Now the histogram of visual word is :")
# x axis has formed by index numbers
x = np.arange(N_cluster) 
#y axis store the count of histogram's kth visual word
y = np.array([abs(np.sum(Histogram[:,h], dtype=np.int32)) for h in range(N_cluster)])

#plotting bar graph
plt.bar(x, y)
plt.xticks(x + 0.5, x)
plt.title("Frequency vs Visual word")
plt.xlabel("Visual Word index")
plt.ylabel("Overall Frequency")
plt.show()


# In[10]:


start=timer()
# Query image
image_path="downloads/Taj Mahal/Image_1.jpeg"
#histogram for query image
input_Histogram = np.zeros(N_cluster)
#reading image
input_image = cv2.imread(image_path)
#normalizing image
input_image = cv2.normalize(input_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#extracting features (keypoint and descriptor of query image)
keyp, descrip = feature_sift.detectAndCompute(input_image,None) 
for j in range(len(descrip)):
    desc_1d = descrip[j]   
    pr =K_Mean.predict([desc_1d])
    input_Histogram[pr] += 1
#predicting nearest images 
Near_neighbour = NearestNeighbors(n_neighbors = 4)
Near_neighbour.fit(Histogram)
#finding index of nearest images
Norm, output_image_index = Near_neighbour.kneighbors([input_Histogram])

#plotting input image
print("Input image :")
imag = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
plt.imshow(imag)
plt.show()

#plotting output image
print("Output images and their ranks\n")
for i in range (len(output_image_index[0])):
    print("Rank:", i+1)
    j=output_image_index[0][i]
    imgg = cv2.cvtColor(image_list[j], cv2.COLOR_BGR2RGB)
    plt.imshow(imgg)
    plt.show()
    print(tag_list[j])

end=timer()
print("Testing time is :",end-start)


# In[ ]:





# In[ ]:




