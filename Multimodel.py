#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tabulate')

import cv2 
import matplotlib.pyplot as plt
import os
from PIL import Image

import numpy as np
import operator
import pandas as pd
import sklearn
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import nltk 
nltk.download('popular')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import math
from gensim.models import FastText
from numpy import dot
from numpy.linalg import norm
from tabulate import tabulate
import time
from timeit import default_timer as timer

if (False):
    # reading query image
    img1 = cv2.imread('bownloads/Albert Einstein/Image_5.jpeg') 
    # Normalizing the image
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    # Creating sift object
    sift = cv2.SIFT_create()
    # Finding keypoints and decriptors for the query image
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    
    j = 0
    images,keypoint_list, descriptor_list = [],[],[]
    for folder in os.listdir("bownloads"):
        if (j == 20):
            break
        for image in os.listdir("bownloads/"+folder):
            # Reading Images from the dataset
            img = cv2.imread(os.path.join("bownloads",folder,image))
            # Normalizing the image
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Finding keypoints and decriptors for image using sift algorithm
            keypoints_2, descriptors_2 = sift.detectAndCompute(img,None)
            keypoint_list.append(keypoints_2)
            descriptor_list.append(descriptors_2)
            images.append(img)
        j += 1
    # match length stores matches and images for the corresponding query image with other images
    match_lengths = []
    # Reading image folder
    i = 0
    j = 0
    start = timer()
    for folder in os.listdir("bownloads"):
        if (j == 20):
            break
        for image in os.listdir("bownloads/"+folder):
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

            threshold = 50
            # if # of good matches > threshold append to match_lengths
            if len(filter_match)>threshold:
                match_lengths.append([filter_match,img,keypoints_2,descriptors_2,folder])
            i += 1
        j += 1

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

else:
    df = pd.read_excel('N_words.xlsx')
    df["url"] = " "
    df["wiki_text"] = " "
    for i in range(len(df)):
        temp = df.loc[i].at["Page"]
        temp = re.sub('\s+', '_', temp)
        # print(temp)
        y = 'https://en.wikipedia.org/wiki/' + temp
        source = urlopen(y)
        df.loc[i].at["url"] = y
        source = source.read()
        soup = BeautifulSoup(source,'lxml')
        paras = []
        for paragraph in soup.find_all('p'):
            paras.append(str(paragraph.text))
        heads = []
        for head in soup.find_all('span', attrs={'mw-headline'}):
            heads.append(str(head.text))
        text = [val for pair in zip(paras, heads) for val in pair]
        text = ' '.join(text)
        text = re.sub(r"\[.*?\]+", '', text)
        text = text.replace('\n', '')[:-11]
        df.loc[i].at["wiki_text"] = text




    df['wiki_text']=[entry.lower() for entry in df['wiki_text']]

    df.wiki_text =df.wiki_text.replace(to_replace='from:(.*\n)',value='',regex=True) #remove from to email 
    df.wiki_text =df.wiki_text.replace(to_replace='lines:(.*\n)',value='',regex=True)
    df.wiki_text =df.wiki_text.replace(to_replace='[!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~]',value=' ',regex=True) #remove punctuation except
    df.wiki_text =df.wiki_text.replace(to_replace='-',value=' ',regex=True)
    df.wiki_text =df.wiki_text.replace(to_replace='\s+',value=' ',regex=True)    #remove new line
    df.wiki_text =df.wiki_text.replace(to_replace='  ',value='',regex=True)                #remove double white space
    df.wiki_text =df.wiki_text.apply(lambda x:x.strip())  # Ltrim and Rtrim of whitespace





    ## ## Checking and drop empty data
    for i,sen in enumerate(df.wiki_text):
        if len(sen.strip()) ==0:
            print(str(i))
            #file_data.text[i] = np.nan
            df=df.drop(str(i),axis=0).reset_index().drop('index',axis=1)

          
          
    # Tokenization: breaking down data into set of words
    df['Word tokenize']= [word_tokenize(entry) for entry in df.wiki_text]





    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    def wordLemmatizer(data):
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        file_clean_k =pd.DataFrame()
        for index,entry in enumerate(data):
          
            # Declaring Empty List to store the words that follow the rules for this step
            Final_words = []
            # Initializing WordNetLemmatizer()
            word_Lemmatized = WordNetLemmatizer()
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(entry):
                # Below condition is to check for Stop words and consider only alphabets
                if len(word)>1 and word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words.append(word_Final)
                    # The final processed set of words for each iteration will be stored in 'text_final'
                    file_clean_k.loc[index,'Keyword_final'] = str(Final_words)
                    file_clean_k.loc[index,'Keyword_final'] = str(Final_words)
        return file_clean_k



    df_clean = wordLemmatizer(df['Word tokenize']) 




    df_clean=df_clean.replace(to_replace ="\[.", value = '', regex = True)
    df_clean=df_clean.replace(to_replace ="'", value = '', regex = True)
    df_clean=df_clean.replace(to_replace =" ", value = '', regex = True)
    df_clean=df_clean.replace(to_replace ='\]', value = '', regex = True)




    ## Insert New column in df_news to stored the Clean Keyword
    df.insert(loc=3, column='Clean_Keyword', value=df_clean['Keyword_final'].tolist())




    df_save= df
    df_save = df_save.drop(['Word tokenize','Clean_Keyword'],axis=1)

    df_save.to_csv("df_index.csv", index=False, header=True)


    def cosine_sim(a, b):
        cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return cos_sim


    # Create the list of list format of the custom corpus for gensim modeling 
    clean_doc_words = [row.split(',') for row in df_clean['Keyword_final']]
    # show the example of list of list format of the custom corpus for gensim modeling 
    # clean_doc_words[:2] 



    model_fast = FastText(clean_doc_words,vector_size=200,min_count=1)



    # import pprint
    def cosine_distance (model, word, df):
        cosine_dict ={}
        cosine_dict_terms = {}
        word_list = []
        term_list = []
        a = model.wv[word]
        # print(df.loc[0].at['Clean_Keyword'])
        for i in range(len(df)):
            max_sim = -2
            temp = str(df.loc[i].at['Clean_Keyword'])
            temp = temp.split(',')
            # print(temp)
            for item in temp:
                b = model.wv[item]
                cos_sim = dot(a.T, b.T)/(norm(a)*norm(b))
                if item not in cosine_dict_terms:
                    cosine_dict_terms[item] = cos_sim
                else:
                    if cosine_dict_terms[item] < cos_sim:
                        cosine_dict_terms[item] = cos_sim
                if cos_sim > max_sim:
                    max_sim = cos_sim
            cosine_dict[df.loc[i].at['Page']] = max_sim
        dist_sort = sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
        dist_term_sort = sorted(cosine_dict_terms.items(),key=lambda d: d[1], reverse=True) 
        for item in dist_sort:
            word_list.append((item[0], item[1]))
        for terms in dist_term_sort:
            term_list.append((terms[0],terms[1]))
        return (word_list,term_list)
    
    
    query = input('Input query:')
    start = timer()
    preprocessed_query = re.sub("\W+", " ", query).strip()
    tokens = word_tokenize(str(preprocessed_query))
    q_df = pd.DataFrame(columns=['q_clean'])
    q_df.loc[0,'q_clean'] =tokens
    q_df['q_clean'] =wordLemmatizer(q_df.q_clean)
    q_df=q_df.replace(to_replace ="\[.", value = '', regex = True)
    q_df=q_df.replace(to_replace ="'", value = '', regex = True)
    q_df=q_df.replace(to_replace =" ", value = '', regex = True)
    q_df=q_df.replace(to_replace ='\]', value = '', regex = True)
    q = pd.DataFrame(columns=['a'])
    q.a = q_df.q_clean



    print('Word2Vec:')
    print('Time taken to make the search:')
    get_ipython().run_line_magic('time', 'word_list,term_list = cosine_distance(model_fast,query,df)')


    y = len(df)
    print('Words closest to given word ranked:')
    for i in range(5):
        print((i+1,word_list[i]))
        for j in range(1,4):
            img = cv2.imread("bownloads/"+word_list[i][0]+"/Image_"+str(j)+".jpeg")
            img_new = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_new,'gray')
            plt.show()

    end = timer()
    print("TIME_TAKEN =",end-start)


# In[ ]:




