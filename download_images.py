#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Import necessary libraries
import pandas as pd
from PIL import Image
from google_images_download import google_images_download
import os
from io import BytesIO
import cairosvg

#Reading the csv file of the set of words
df = pd.read_excel('Book1.xlsx')

# loop through the set of top 100 words
for i in range(len(df)):
    if (i >= 19):
        query_string = df.loc[i].at["Page"] # word i

        # download google images using the word as a keyword
        response = google_images_download.googleimagesdownload()

        # arguments for the google image downloader
        arguments = {"keywords":query_string,"limit":55,"format":"jpg"}
        # path where images are saved
        paths = response.download(arguments)

        # go to the saved images folder
        path = "C:/Users/LENOVO/Desktop/ELL786/downloads/"+query_string
        folder = os.listdir(path)
        j = 0
        for images in folder:
            j += 1
            if (images.split('.')[-1] == 'svg'):
                # image extension is an SVG so we use cairosvg to convert it to png and read it using PIL
                out = BytesIO()
                cairosvg.svg2png(url=path+"/"+images, write_to=out)
                image = Image.open(out)
            else:
                # else we simply read the image using PIL
                image = Image.open(path+"/"+images)
            # resize the image to 1000x1000 preserving the aspect ratio
            image.thumbnail(size=(1000,1000))
            if image.mode != 'RGB':
                # convert to RGB if its not already in that format
                image = image.convert('RGB')
            # save image
            image.save(path+'\Image_'+str(j)+'.jpeg', optimize=True, quality=65)
            os.remove(path+"/"+images)

        
        


# In[ ]:




