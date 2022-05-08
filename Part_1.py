#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install wikipedia
get_ipython().system('pip install sklearn')
get_ipython().system('pip install gensim')
get_ipython().system('pip install --upgrade nltk')


# In[2]:


# import wikipedia
import numpy as np
import operator
import pandas as pd
import sklearn
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import nltk 
nltk.download('popular')
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import math

from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[3]:


df = pd.read_excel('/content/drive/My Drive/ELL_786/N_words.xlsx')
df["url"] = " "
df["wiki_text"] = " "
for i in range(len(df)):
  temp = df.loc[i].at["Page"]
  # for j in range(len(temp)):
  #   if(temp[j] == ' '):
  #     tmp = list(temp)
  #     tmp[j] = '_'
  #     temp = ''.join(tmp)
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

df


# In[4]:


df['wiki_text']=[entry.lower() for entry in df['wiki_text']]

df.wiki_text =df.wiki_text.replace(to_replace='from:(.*\n)',value='',regex=True) #remove from to email 
df.wiki_text =df.wiki_text.replace(to_replace='lines:(.*\n)',value='',regex=True)
df.wiki_text =df.wiki_text.replace(to_replace='[!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~]',value=' ',regex=True) #remove punctuation except
df.wiki_text =df.wiki_text.replace(to_replace='-',value=' ',regex=True)
df.wiki_text =df.wiki_text.replace(to_replace='\s+',value=' ',regex=True)    #remove new line
df.wiki_text =df.wiki_text.replace(to_replace='  ',value='',regex=True)                #remove double white space
df.wiki_text =df.wiki_text.apply(lambda x:x.strip())  # Ltrim and Rtrim of whitespace

df


# In[5]:


## ## Checking and drop empty data
for i,sen in enumerate(df.wiki_text):
    if len(sen.strip()) ==0:
        print(str(i))
        #file_data.text[i] = np.nan
        df=df.drop(str(i),axis=0).reset_index().drop('index',axis=1)


# In[6]:


# Tokenization: breaking down data into set of words
df = df.sample(n=20) ## taking 20 samples out of the given N_words
df.reset_index(inplace=True,drop=True)
df['Word tokenize']= [word_tokenize(entry) for entry in df.wiki_text]

df


# In[ ]:


# from nltk.stem import PorterStemmer
# ps = PorterStemmer()
# # print(df.loc[0].at["Word tokenize"])
# for i in range(len(df)):
#   for j in range(len(df.loc[i].at["Word tokenize"])):
#     df.loc[i].at['Word tokenize'][j] = ps.stem(df.loc[i].at['Word tokenize'][j])
# print(df.loc[0].at['Word tokenize'])


# In[7]:


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


# In[8]:


df_clean = wordLemmatizer(df['Word tokenize']) 
df_clean


# In[9]:


df_clean=df_clean.replace(to_replace ="\[.", value = '', regex = True)
df_clean=df_clean.replace(to_replace ="'", value = '', regex = True)
df_clean=df_clean.replace(to_replace =" ", value = '', regex = True)
df_clean=df_clean.replace(to_replace ='\]', value = '', regex = True)


# In[10]:


## Insert New column in df_news to stored the Clean Keyword
df.insert(loc=3, column='Clean_Keyword', value=df_clean['Keyword_final'].tolist())
df


# In[11]:


df_save= df
df_save = df_save.drop(['Word tokenize','Clean_Keyword'],axis=1)
df_save


# In[12]:


df_save.to_csv("df_index.csv", index=False, header=True)


# ### Search Engine Using TF - IDF

# In[13]:


def gen_vector_T(tokens):
    Q = np.zeros((len(vocabulary)))
    x= tfidf.transform(tokens)
    for token in tokens[0].split(','):
        try:
            ind = vocabulary.index(token)
            Q[ind]  = x[0, tfidf.vocabulary_[token]]
        except:
            pass
    return Q


# In[14]:


def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


# In[15]:


## Create Vocabulary
vocabulary = set()

for doc in df.Clean_Keyword:
  vocabulary.update(doc.split(','))

vocabulary = list(vocabulary)

# Intializating the tfIdf model
tfidf = TfidfVectorizer(vocabulary=vocabulary,dtype=np.float32)

# Fit the TfIdf model
tfidf.fit(df.Clean_Keyword)

# Transform the TfIdf model
tfidf_tran=tfidf.transform(df.Clean_Keyword)

vocab_vector = []
for i in range(len(vocabulary)):
  vec = gen_vector_T([vocabulary[i]])
  vocab_vector.append(vec)


# In[16]:


ti_vectors = [[0]*len(df)]*len(vocabulary)
for i in range(len(vocabulary)):
  d_cosines = []
  for d in tfidf_tran.A:
    d_cosines.append(cosine_sim(vocab_vector[i], d))
  ti_vectors[i] = d_cosines 


# In[17]:


import pickle


# In[ ]:


### Save model
with open('/content/drive/My Drive/ELL_786/tfid.pkl','wb') as handle:
    pickle.dump(tfidf_tran, handle)


# In[ ]:


### load model
t = pickle.load(open('/content/drive/My Drive/ELL_786/tfid.pkl','rb'))


# In[ ]:


### Save Vocabulary
with open("/content/drive/My Drive/ELL_786/vocabulary_top_100_list.txt", "w") as file:
    file.write(str(vocabulary))


# In[ ]:


### load Vocabulary
with open("/content/drive/My Drive/ELL_786/vocabulary_top_100_list.txt", "r") as file:
    data2 = eval(file.readline())


# In[ ]:


# data2


# In[18]:


def cosine_similarity_T(k, q):
    #print("Cosine Similarity")
    query_vector = gen_vector_T(q['a'])
    d_cosines = []
  
    for d in tfidf_tran.A:
      d_cosines.append(cosine_sim(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]
    d_cosines.sort()
    a = pd.DataFrame()
    for i,index in enumerate(out):
        a.loc[i,'index'] = str(index)
        a.loc[i,'Page'] = df['Page'][index]
        a.loc[i,'url'] = df['url'][index]
    for j,simScore in enumerate(d_cosines[-k:][::-1]):
        a.loc[j,'Score'] = simScore
    return a


# In[19]:


def cosine_sim_closest_terms(q):
  query_vector = gen_vector_T(q['a'])
  dist = np.zeros((len(vocabulary),2))
  d_cosines = []
  for d in tfidf_tran.A:
    d_cosines.append(cosine_sim(query_vector, d))
  for i in range(len(vocabulary)):
    dist[i,0] = cosine_sim(d_cosines,ti_vectors[i])
    dist[i,1] = i
  # print(dist)
  np.sort(dist)[::-1]
  # print('10 closest search terms to the query word:')
  clo_terms = []
  for i in range(len(vocabulary)):
    clo_terms.append(vocabulary[int(dist[i,1])])
  return clo_terms


# In[ ]:


# query = input('Type the query word to search: ')
# y = len(df)
# %time cosine_similarity_T(y,query)


# ### Word2vec

# In[20]:


# Create the list of list format of the custom corpus for gensim modeling 
clean_doc_words = [row.split(',') for row in df_clean['Keyword_final']]
# show the example of list of list format of the custom corpus for gensim modeling 
# clean_doc_words[:2] 


# In[ ]:


## Train the genisim word2vec model with our own custom corpus
model = Word2Vec(clean_doc_words, size=200,min_count=1,workers=3, sg = 1)


# In[21]:


from gensim.models import FastText
model_fast = FastText(clean_doc_words,size=200,min_count=1)


# In[23]:


from numpy import dot
from numpy.linalg import norm
# import pprint
def cosine_distance (model, word, df):
    cosine_dict ={}
    cosine_dict_terms = {}
    word_list = []
    term_list = []
    a = model[word]
    # print(df.loc[0].at['Clean_Keyword'])
    for i in range(len(df)):
      max_sim = -2
      temp = str(df.loc[i].at['Clean_Keyword'])
      temp = temp.split(',')
      # print(temp)
      for item in temp:
        b = model [item]
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


# In[45]:


get_ipython().run_cell_magic('capture', 'cap --no-stderr', 'from tabulate import tabulate\ndf_test = pd.read_excel(\'/content/drive/My Drive/ELL_786/Test_Set.xlsx\')\nprint(\'Assignment 3 part 1:\')\nprint(\'Artcles taken: 101\')\nprint()\nrandom_list_30_words = df_test[\'Page\'].tolist()\n# print(random_list_30_words)\n# random_list_30_words = {\'majestic\',\'grate\',\'strengthen\',\'pies\',\'loving,run\',\'bike\',\'old-fashioned\',\'moan\',\'park\',\'fantastic\',\'aunt\',\'brief\',\'jolly\',\'rhetorical\',\'rob\',\'clever\',\'ambiguous\',\'dusty\',\'haircut\',\'beneficial\',\'true\',\'foot\',\'license\',\'languid\',\'boundless\',\'rebel\',\'placid\',\'look\',\'exciting\'}\n# query = input(\'Input query:\')\nfor i in range(len(random_list_30_words)):\n  query = random_list_30_words[i]\n  print(\'query: \',query)\n  preprocessed_query = re.sub("\\W+", " ", query).strip()\n  tokens = word_tokenize(str(preprocessed_query))\n  q_df = pd.DataFrame(columns=[\'q_clean\'])\n  q_df.loc[0,\'q_clean\'] = tokens\n  q_df[\'q_clean\'] = wordLemmatizer(q_df.q_clean)\n  q_df=q_df.replace(to_replace ="\\[.", value = \'\', regex = True)\n  q_df=q_df.replace(to_replace ="\'", value = \'\', regex = True)\n  q_df=q_df.replace(to_replace =" ", value = \'\', regex = True)\n  q_df=q_df.replace(to_replace =\'\\]\', value = \'\', regex = True)\n  q = pd.DataFrame(columns=[\'a\'])\n  q.a = q_df.q_clean\n  print(\'TF-IDF:\')\n  y = len(df)\n\n  print(\'Time taken to make the search:\')\n  %time aux = cosine_similarity_T(y,q)\n\n  print()\n  print(\'Articles closest to given word ranked:\')\n  for i in range(y):\n    print((i+1,aux.loc[i].at["Page"]))\n  print()\n  print(\'10 closest search terms to the query:\')\n  closest_terms_ti = cosine_sim_closest_terms(q)\n  for i in range(10):\n    print(i+1,closest_terms_ti[i])\n  print()\n\n  print(\'Word2Vec:\')\n  print(\'Time taken to make the search:\')\n  %time word_list,term_list = cosine_distance(model_fast,query,df)\n\n  print(\'Articles closest to given word ranked:\')\n  for i in range(y):\n    print((i+1,word_list[i]))\n  print()\n  print(\'10 closest search terms to the query:\')\n  for i in range(10):\n    print(term_list[i])\n  print()\n  print(\'Rank in Word2Vec of first 10 closest articles in tf-idf to query:\')\n  for i in range(10):\n    temp = aux.loc[i].at["Page"]\n    print(\'Article:\',temp)\n    print(\'Rank in tf-idf:\',str(i+1))\n    tmp = 0\n    for j in range(len(word_list)):\n      if(word_list[j][0] == temp):\n        tmp = j+1\n        break\n    print(\'Rank in Word2Vec:\',str(tmp))\n  print()\n\n  if(i < 15):\n   print(\'Selected Algo: tf-idf\')\n   algo = 1\n  else:\n    print(\'Selected Algo: word2vec\')\n    algo = 2\n  \n  # print(\'Input 1 for tf-idf and 2 for word2vec\')\n  \n  # print(\'List of Articles:\')\n  # for i in range(y):\n  #   print((i+1,df.loc[i].at[\'Page\']))\n  # print(\'Select an article by giving the corresponding number\')\n  article = 0\n  if(i < 15):\n    article = i+1\n    print(\'Article selected: \',df.loc[article-1].at[\'Page\'])\n  else:\n    article = i-10\n  article = int(article)\n  print()\n  # print(\'Article chosen:\',df.loc[article-1].at[\'Page\'])\n  # print()\n  print(\'link of the article:\')\n  print(df.loc[article-1].at[\'url\'])\n  print()\n  tr = 0\n  for i in range(y):\n    if df.loc[article-1].at[\'Page\'] == aux.loc[i].at[\'Page\']:\n      tr = i+1\n      break\n  print(\'tf-idf rank of article with respect to query:\',str(tr))\n  wv_rank = 0\n  for i in range(y):\n    if df.loc[article-1].at[\'Page\'] == word_list[i][0]:\n      wv_rank = i+1\n      break\n  print(\'Word2Vec rank of article with respect to query:\',str(wv_rank))\n  print()\n  if(int(algo) == 1):\n    print(\'10 Closest Search terms using tf-idf of the selected article: \')\n    j = 0\n    for i in range(len(vocabulary)):\n      if(j<10):\n        if(closest_terms_ti[i] in df.loc[article-1].at[\'Clean_Keyword\']):\n          j += 1\n          print(j,closest_terms_ti[i])\n      else:\n        break\n  else:\n    print(\'10 Closest Search terms using word2vec of the selected article: \')\n    j = 0\n    for i in range(len(vocabulary)):\n      if(j<10):\n        if(term_list[i][0] in df.loc[article-1].at[\'Clean_Keyword\']):\n          j += 1\n          print(j,term_list[i][0])\n      else:\n        break\n  print()\n  with open(\'/content/drive/My Drive/ELL_786/results_on_30_words.txt\',\'w\') as f:\n    f.write(cap.stdout)\n  print()')

