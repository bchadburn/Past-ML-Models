import requests
import pandas as pd
import numpy as np
#from WebFocusedCrawl.items import WebfocusedcrawlItem  # item class
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup 
import nltk  # used to remove stopwords from tags
import re  # regular expressions used to parse tags
import urllib3
import json
import time
from nltk.corpus import stopwords
import string
from collections import Counter
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

search_urls = ['https://www.indeed.com/q-data-scientist-l-united-states-jobs.html','https://www.indeed.com/jobs?q=data + scientist&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=data + scientist&l=United+States&start=20', 'https://www.indeed.com/jobs?q=data + scientist&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=data + scientist&l=United+States&start=40','https://www.indeed.com/jobs?q=data + scientist&l=United+States&start=50', 
               'https://www.indeed.com/q-machine-learning-l-united-states-jobs.html','https://www.indeed.com/jobs?q=machine+learning&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=machine+learning&l=United+States&start=20', 'https://www.indeed.com/jobs?q=machine+learning&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=machine+learning&l=United+States&start=40','https://www.indeed.com/jobs?q=machine+learning&l=United+States&start=50']
# Try
#http = urllib3.PoolManager(
#...     cert_reqs='CERT_REQUIRED',
#...     ca_certs=certifi.where())

urls = []
for link in search_urls:
    http = urllib3.PoolManager()
    response = http.request('GET', link)
    soup = BeautifulSoup(response.data)
    for link in soup.find_all('div', class_='title'):
        partial_url = link.a.get('href')
        url = 'https://indeed.com' + partial_url
        if url not in urls:
            urls.append(url)

# urls_reduced = [i for i in urls if i.startswith(('https://indeed.com/rc','https://indeed.com/company'))]  # Filter out sponsered links

data = []
item = {} 
j = 0 
for i in urls:
    html = requests.get(urls[j], time.sleep(1)).content
    unicode_str = html.decode("utf8")
    encoded_str = unicode_str.encode("ascii",'ignore')
    soup = BeautifulSoup(encoded_str, "html.parser")
    if len(soup.find_all("h3")) == 0: # This ensures the element h3 exists to avoid returning error.
        j = j + 1
        continue
    item = {}
    item['ID'] = j
    item['URL'] = i
    item['TITLE'] = soup.find("h3", {"class": "icl-u-xs-mb--xs icl-u-xs-mt--none jobsearch-JobInfoHeader-title"}).text
    item['BODY'] = soup.find("div", {"class": "jobsearch-jobDescriptionText"}).text
    data.append(item) # add the item to the list
    j = j+1

# Although we required unique urls we still have duplicate postings     
data = list({v['BODY']:v for v in data}.values())
        
with open('jobopenings.json', 'w') as fp:
    json.dump(data, fp)

############# Cleaning data ############
# Load stop words and add new stop words
stop_words = stopwords.words('english')
newStopWords = ['Jr','Junior', 'junior', 'Senior', 'Sn', 'Lead', 'Entry', 
                'Level', 'Internship', 'Internships','Intern', 'Summer', 
                'exp req' ]
stop_words.extend(newStopWords)

def clean_doc(doc):
# split into tokens by white space
    tokens = []
    titles = []
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    for d in doc:
        titles = d['TITLE']   
        titles = titles.split()
        # prepare regex for char filtering
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        # remove punctuation from each word
        titles = [re_punc.sub('', w) for w in titles]
        # remove remaining titles that are not alphabetic
        titles = [word for word in titles if word.isalpha()]
        # filter out stop words
        titles = [w for w in titles if not w in stop_words]
        # filter out short titles
        titles = [word for word in titles if len(word) > 1]
        # join words back to re-create title
        titles = [' '.join(word for word in titles)]
        tokens.append(titles)
    return tokens

tokens = clean_doc(data)
# I've created a list of list of strings, flattening to list of strings
flattened_list = []
for x in tokens:
    for y in x:
        flattened_list.append(y)
      
tokens = flattened_list

# most common title
Counter(tokens).most_common(10)

sets_of_words.append(tokens[0].split()[1])

# Creating list of bigrams for frequency count
sets_of_words = []
for i in range(0, len(tokens)):
    length = len(tokens[i].split())
    j = 0
    for j in range(0, length-1):
        titles = ()
        titles = (tokens[i].split()[j], tokens[i].split()[j+1])
        tokens[i].split()[j]
        sets_of_words.append(titles)

# Creating list of word trigrams for frequency count
for i in range(0, len(tokens)):
    length = len(tokens[i].split())
    j = 0
    for j in range(0, length-2):
        titles = ()
        titles = (tokens[i].split()[j], tokens[i].split()[j+1], tokens[i].split()[j+2])
        tokens[i].split()[j]
        sets_of_words.append(titles)
        
Counter(sets_of_words).most_common(10)   

# We find the most frequency titles are:
# 1. Machine Learning Engineer (20.8%). 2. Data Scientist(16%).
# 3. Research Scientist (7%)
# 4. Machine learning Developer (6.4%)
# 5. Machine Learning Scientist (6.4%)

################## Clean the body/text of the job openings ##################


stop_words = stopwords.words('english')
newStopWords = ['abilities','Abilities', 'ability', 'Ability','able', 'Able', 
                'employer', 'opportunity', 'race', 'color', 'orientation', 
                'gender', 'sex', 'sexual']
stop_words.extend(newStopWords)

# Code to keep strings seperate
token_strings = []
def clean_doc(doc):
# split into tokens by white space   
    i = 0
    while i < len(doc):
        # prepare regex for char filtering
        text = doc[i].split()
        # prepare regex for char filtering
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        text = [re.sub("-"," ",w) for w in text]
        # remove punctuation from each word
        text = [re_punc.sub('', w) for w in text]
        # remove remaining titles that are not alphabetic
        text = [word for word in text if word.isalpha()]
        # filter out stop words
        text = [w for w in text if not w in stop_words]
        # filter out short titles
        text = [word for word in text if len(word) > 1]
        # join words back to re-create title
        text = [' '.join(word for word in text)]
        token_strings.append(text)
        i = i + 1
    return token_strings

clean_doc(text)

flattened_list = []
for x in token_strings:
    for y in x:
        flattened_list.append(y)
      
token_strings = flattened_list

# Reduce to string and split words
tokens = tokens[0][0].split()
# most common words
Counter(tokens).most_common(10)

################## Using TF-IDF to find words to search on Indeed ############

# return bigrams
bgs = nltk.bigrams(tokens)
#compute frequency distribution for all the bigrams in the text
fdist = nltk.FreqDist(bgs)
print(fdist.most_common(10))
 
# return trigrams
tgs = nltk.trigrams(tokens)
fdist_tg = nltk.FreqDist(tgs)
print(fdist_tg.most_common(10))

# TF-IDF
tokens1 = [' '.join(word for word in tokens)]

#https://stackoverflow.com/questions/25217510/how-to-see-top-n-entries-of-term-document-matrix-after-tfidf-in-scikit-learn/25219535
vectorizer = TfidfVectorizer(ngram_range=(1,3))
X = vectorizer.fit_transform(token_strings)
features_by_gram = defaultdict(list)
for f, w in zip(vectorizer.get_feature_names(), vectorizer.idf_):
    features_by_gram[len(f.split(' '))].append((f, w))
top_n = 5
allfeatures = []
for gram, features in features_by_gram.items():
    top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
    top_features = [f[0] for f in top_features]
    allfeatures.append(top_features)
    print ('{}-gram top:'.format(gram), top_features)
    
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(token_strings)
# summarize
print(vectorizer.vocabulary_)
vectorizer.vocabulary_

import numpy as np
tfidf = TfidfVectorizer()
string1 = 'tree cat travellers fruit jupiter'
response = tfidf.transform([string1])
feature_array = np.array(tfidf.get_feature_names())
tfidf_sorting = np.argsort(response.toarray(tokens)).flatten()[::-1]

n = 3
top_n = feature_array[tfidf_sorting][:n]

# We find the following useful terms based on frequency of bigrams and trigrams

# Word frequency
#[('data', 333),
# ('learning', 250),
# ('machine', 174),
# ('experience', 151),
# ('team', 117),
# ('work', 112),
# ('We', 108),
# ('Experience', 92),
# ('development', 88),
# ('models', 83)]

#bigrams
#(('Machine', 'Learning'), 83),
# (('Learning', 'Engineer'), 27),
# (('Data', 'Scientist'), 20),
# (('Learning', 'Intern'), 12),
# (('Scientist', 'Machine'), 10),

#trigrams
#(('Machine', 'Learning', 'Engineer'), 26),
# (('Machine', 'Learning', 'Intern'), 12),
# (('Data', 'Scientist', 'Data'), 9),
# (('Scientist', 'Machine', 'Learning'), 9),
# (('Machine', 'Learning', 'Developer'), 8),
# (('Engineer', 'Machine', 'Learning'), 8),
# (('Machine', 'Learning', 'Scientist'), 8),

# TFIDF
#1-gram top: ['acg', 'ai', 'aimachine', 'algorithms', 'analysis']
#2-gram top: ['acg machine', 'ai engineering', 'ai machine', 'aimachine learningml', 'algorithms statistics']
#3-gram top: ['acg machine learning', 'ai engineering machine', 'ai machine learning', 'aimachine learningml technical', 'algorithms statistics machine']

# TF-IDF keeping documents seperate
#1-gram top: ['abreast', 'abuse', 'academics', 'academicstrong', 'accepted']
#2-gram top: ['ab testing', 'ab tests', 'about capgemini', 'about deepcure', 'about job']
#3-gram top: ['ab testing experimentation', 'ab tests automate', 'about capgemini with', 'about deepcure deepcure', 'about job google']

fdist.most_common(10)
fdist_tg.most_common(10)

#allfeatures
#new_urls = []
#firstword = "machine"
#secondword = "learning"
#new_url = 'https://www.indeed.com/q-{}-{}-l-united-states-jobs.html'.format(firstword,secondword)

# Initial Urls to being scrapping. 
search_urls = ['https://www.indeed.com/q-data-scientist-l-united-states-jobs.html','https://www.indeed.com/jobs?q=data+scientist&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=data+scientist&l=United+States&start=20', 'https://www.indeed.com/jobs?q=data+scientist&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=data+scientist&l=United+States&start=40','https://www.indeed.com/jobs?q=data+scientist&l=United+States&start=50', 
               'https://www.indeed.com/q-machine-learning-l-united-states-jobs.html','https://www.indeed.com/jobs?q=machine+learning&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=machine+learning&l=United+States&start=20', 'https://www.indeed.com/jobs?q=machine+learning&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=machine+learning&l=United+States&start=40','https://www.indeed.com/jobs?q=machine+learning&l=United+States&start=50'
               'https://www.indeed.com/q-deep-learning-l-united-states-jobs.html','https://www.indeed.com/jobs?q=deep+learning&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=deep+learning&l=United+States&start=20', 'https://www.indeed.com/jobs?q=deep+learning&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=deep+learning&l=United+States&start=40','https://www.indeed.com/jobs?q=deep+learning&l=United+States&start=50',
               'https://www.indeed.com/q-learning-algorithms-l-united-states-jobs.html','https://www.indeed.com/jobs?q=learning+algorithms&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=learning+algorithms&l=United+States&start=20', 'https://www.indeed.com/jobs?q=learning+algorithms&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=learning+algorithms&l=United+States&start=40','https://www.indeed.com/jobs?q=learning+algorithms&l=United+States&start=50',
               'https://www.indeed.com/q-machine-learning-algorithms-l-united-states-jobs.html','https://www.indeed.com/jobs?q=machine+learning&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=machine+learning+algorithms&l=United+States&start=20', 'https://www.indeed.com/jobs?q=machine+learning+algorithms&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=machine+learning+algorithms&l=United+States&start=40','https://www.indeed.com/jobs?q=machine+learning+algorithms&l=United+States&start=50',
               'https://www.indeed.com/q-natural-language-processing-l-united-states-jobs.html','https://www.indeed.com/jobs?q=natural+language+processing&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=natural+language+processing&l=United+States&start=20', 'https://www.indeed.com/jobs?q=natural+language+processing&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=natural+language+processing&l=United+States&start=40','https://www.indeed.com/jobs?q=natural+language+processing&l=United+States&start=50',
               'https://www.indeed.com/q-statistics-machine-learning-l-united-states-jobs.html','https://www.indeed.com/jobs?q=statistics+machine+learning&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=statistics+machine+learning&l=United+States&start=20', 'https://www.indeed.com/jobs?q=statistics+machine+learning&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=statistics+machine+learning&l=United+States&start=40','https://www.indeed.com/jobs?q=statistics+machine+learning&l=United+States&start=50',
               'https://www.indeed.com/q-machine-learning-engineer-l-united-states-jobs.html','https://www.indeed.com/jobs?q=machine+learning+engineer&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=machine+learning+engineer&l=United+States&start=20', 'https://www.indeed.com/jobs?q=machine+learning+engineer&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=machine+learning+engineer&l=United+States&start=40','https://www.indeed.com/jobs?q=machine+learning+engineer&l=United+States&start=50',
               'https://www.indeed.com/q-learning-deep-learning-l-united-states-jobs.html','https://www.indeed.com/jobs?q=learning+deep+learning&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=learning+deep+learning&l=United+States&start=20', 'https://www.indeed.com/jobs?q=learning+deep+learning&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=learning+deep+learning&l=United+States&start=40','https://www.indeed.com/jobs?q=learning+deep+learning&l=United+States&start=50',
               'https://www.indeed.com/q-data-engineer-l-united-states-jobs.html','https://www.indeed.com/jobs?q=data+engineer&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=data+engineer&l=United+States&start=20', 'https://www.indeed.com/jobs?q=data+engineer&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=data+engineer&l=United+States&start=40','https://www.indeed.com/jobs?q=data+engineer&l=United+States&start=50',
               'https://www.indeed.com/q-data-analyst-l-united-states-jobs.html','https://www.indeed.com/jobs?q=data+analyst&l=United+States&start=10',
               'https://www.indeed.com/jobs?q=data+analyst&l=United+States&start=20', 'https://www.indeed.com/jobs?q=data+analyst&l=United+States&start=30',
               'https://www.indeed.com/jobs?q=data+analyst&l=United+States&start=40','https://www.indeed.com/jobs?q=data+analyst&l=United+States&start=50']

# Scrapping urls
urls = []
for link in search_urls:
    http = urllib3.PoolManager()
    response = http.request('GET', link)
    soup = BeautifulSoup(response.data)
    for link in soup.find_all('div', class_='title'):
        partial_url = link.a.get('href')
        url = 'https://indeed.com' + partial_url
        if url not in urls:
            urls.append(url)

# urls_reduced = [i for i in urls if i.startswith(('https://indeed.com/rc','https://indeed.com/company'))]  # Filter out sponsered links

data = []
item = {} 
j = 0 
for i in urls:
    html = requests.get(urls[j], time.sleep(1)).content
    unicode_str = html.decode("utf8")
    encoded_str = unicode_str.encode("ascii",'ignore')
    soup = BeautifulSoup(encoded_str, "html.parser")
    if len(soup.find_all("h3")) == 0: # This ensures the element h3 exists to avoid returning error.
        j = j + 1
        continue
    item = {}
    item['ID'] = j
    item['URL'] = i
    try:
        item['TITLE'] = soup.find("h3", {"class": "icl-u-xs-mb--xs icl-u-xs-mt--none jobsearch-JobInfoHeader-title"}).text
    except:
        j = j+1
        continue
    try: 
        item['BODY'] = soup.find("div", {"class": "jobsearch-jobDescriptionText"}).text
    except:
        j = j+1
        continue
    data.append(item) # add the item to the list
    j = j+1

# Although we required unique urls we still have duplicate postings     
data = list({v['BODY']:v for v in data}.values())
 
# Saving to JSON, one doc per line
with open('jobopenings.json', 'w') as fp:
    fp.write(
        '[' +
        ',\n'.join(json.dumps(i) for i in data) +
        ']\n')
    
###############################################################################
##### Read in data and train model
"""
Template code: 
Created on Sun May 12 20:15:05 2019
@author: Paul Huynh
"""
# Most of the code is mine.
###############################################################################
RANDOM_SEED = 999      
title={'TITLE':[]}
body={'BODY':[]}
identifier = {'ID':[]}

with open('jobopenings.json') as f:
    data = json.load(f)

for item in data:
    title['TITLE'].append(item['TITLE'])
    body['BODY'].append(item['BODY'])
    identifier['ID'].append(item['ID'])

        
#### Cleaning data
# Load stop words and add new stop words
    
data=pd.concat([pd.DataFrame(title),pd.DataFrame(body),
                pd.DataFrame(identifier)], axis=1)

data=data.reset_index()

###############################################################################
### Function to process documents
###############################################################################
    
stop_words = stopwords.words('english')
newStopWords = ['jr','junior','senior', 'sr', 'lead', 'entry', 'principle',
                'level', 'internship', 'internships','intern', 'summer',  
                'exp']
stop_words.extend(newStopWords)
stop_words = set(stop_words)

def clean_doc(doc):
    # split into tokens by white space
    # prepare regex for char filtering
    tokens = doc.split()
    # remove punctuation from each word
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re.sub("-"," ",w) for w in tokens]
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 2]
    #lowercase all words
    tokens = [word.lower() for word in tokens]
    # filter out stop words
    tokens = [w for w in tokens if not w in stop_words] 
    # word stemming    
    # ps=PorterStemmer()
    # tokens=[ps.stem(word) for word in tokens]        
    return tokens

###############################################################################
### Processing text into lists
###############################################################################

#create empty list to store text documents titles
titles=[]

#for loop which appends the DSI title to the titles list
for i in range(0,len(data)):
    temp_text=data['TITLE'].iloc[i]
    titles.append(temp_text)

#create empty list to store text documents
text_title=[]

#for loop which appends the text to the text_body list
for i in range(0,len(data)):
    temp_title=data['TITLE'].iloc[i]
    text_title.append(temp_title)
    
#empty list to store processed titles
processed_title=[]
#for loop to process the text to the processed_text list
for i in text_title:
    title=clean_doc(i)
    processed_title.append(title)
   
#stitch back together individual words to reform body of text
processed_title_combined=[]
for i in processed_title:
    temp=' '.join(i)
    processed_title_combined.append(temp)
    
########## Body
#create empty list to store text documents
text_body=[]

#for loop which appends the text to the text_body list
for i in range(0,len(data)):
    temp_text=data['BODY'].iloc[i]
    text_body.append(temp_text)

########### Change stop words for documents
stop_words = stopwords.words('english')
newStopWords = ['abilities','Abilities', 'ability', 'Ability','able', 'Able', 
                'employer', 'opportunity', 'race', 'color', 'orientation', 
                'gender', 'sex', 'sexual']
stop_words.extend(newStopWords)
stop_words = set(stop_words)

#empty list to store processed documents
processed_text=[]
#for loop to process the text to the processed_text list
for i in text_body:
    text=clean_doc(i)
    processed_text.append(text)

#Note: the processed_text is the PROCESSED list of documents read directly form 
#the csv.  Note the list of words is separated by commas.

#stitch back together individual words to reform body of text
final_processed_text=[]

for i in processed_text:
    temp_DSI=' '.join(i)
    final_processed_text.append(temp_DSI)


#Note: We stitched the processed text together so the TFIDF vectorizer can work.
#Final section of code has 3 lists used.  2 of which are used for further processing.
#(1) text_body - unused, (2) processed_text (used in W2V), 
#(3) final_processed_text (used in TFIDF), and (4) DSI titles (used in TFIDF Matrix)


######## Reviewing common job titles


# most common words
Counter(processed_title_combined).most_common(10) 

processed_title_combined = [w.replace('science', 'scientist') for w in processed_title_combined]
processed_title_combined = [w.replace('natural language processing', 'machine learning') for w in processed_title_combined]
processed_title_combined = [w.replace('deep learning', 'machine learning') for w in processed_title_combined]
processed_title_combined = [w.replace('big data', 'machine learning') for w in processed_title_combined]
processed_title_combined = [w.replace('business intelligence', 'data analyst') for w in processed_title_combined]
processed_title_combined = [w.replace('database', 'data') for w in processed_title_combined]

title_categories = ['data scientist', 'data engineer', 'machine learning', 'analyst']

# Filter for 4 job titles only


final_processed_titles = []
i=0
while i < len(data):
    j=0
    while j < len(title_categories):
        if title_categories[j] in processed_title_combined[i]:
            final_processed_titles.append(title_categories[j])
            i += 1
            break
        else:
            j += 1
    final_processed_titles.append("other")
    i += 1  
    
data['TITLE'] = final_processed_titles

#### Continue to clean text
def concatenate_list_data(list):
    result= []
    for element in list:
        result += (element)
    return result

# First need to create a list of strings
final_processed_text_strings= []
for d in final_processed_text:
    temporary = d.split()
    final_processed_text_strings.append(temporary)

# Our first approach is simply using the most common words. We then can
# manually eliminate terms we think are too general. 
# This should also be reviewed in order to add additional stop words.

final_processed_text_strings = concatenate_list_data(final_processed_text_strings)

# Adding equivalent classes
final_processed_text_strings = [w.replace('analysis', 'analytics') for w in final_processed_text_strings]
final_processed_text_strings = [w.replace('analysis', 'analytics') for w in final_processed_text_strings]
final_processed_text_strings = [w.replace('statistical', 'statistics') for w in final_processed_text_strings]
final_processed_text_strings = [w.replace('modeling', 'models') for w in final_processed_text_strings]
final_processed_text_strings = [w.replace('models', 'model') for w in final_processed_text_strings]
final_processed_text_strings = [w.replace('scientists', 'scientist') for w in final_processed_text_strings]

freq_terms = Counter(final_processed_text_strings).most_common(100)

###############################################################################
### Sklearn TFIDF 
###############################################################################
#note the ngram_range will allow you to include multiple words within the TFIDF matrix
#Call Tfidf Vectorizer
Tfidf=TfidfVectorizer(ngram_range=(1,1))

#fit the vectorizer using final processed documents.  The vectorizer requires the 
#stiched back together document.

TFIDF_matrix=Tfidf.fit_transform(final_processed_text)     

#creating datafram from TFIDF Matrix
matrix=pd.DataFrame(TFIDF_matrix.toarray(), columns=Tfidf.get_feature_names(), index=titles)

###############################################################################
### Doc2Vec
###############################################################################
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(final_processed_text)]
model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)

doc2vec_df=pd.DataFrame()
for i in range(0,len(processed_text)):
    vector=pd.DataFrame(model.infer_vector(processed_text[i])).transpose()
    doc2vec_df=pd.concat([doc2vec_df,vector], axis=0)

doc2vec_df=doc2vec_df.reset_index()

doc_titles={'title': titles}
t=pd.DataFrame(doc_titles)

doc2vec_df=pd.concat([doc2vec_df,t], axis=1)

doc2vec_df=doc2vec_df.drop('index', axis=1)


###############################################################################
### K Means Clustering - TFIDF
###############################################################################
# We using kmeans and TSNE (multidimesional scaling) on documents. To use it for
# terms just transform the matrix: matrix=matrix.transpose() 
k=5
km = KMeans(n_clusters=k, random_state =RANDOM_SEED)
km.fit(TFIDF_matrix)
clusters = km.labels_.tolist()

terms = Tfidf.get_feature_names()
Dictionary={'ID':list(data["ID"]), 'Cluster':clusters,  'Text': final_processed_text}
frame=pd.DataFrame(Dictionary, columns=['Cluster', 'ID','Text'])


frame=pd.concat([frame,data['TITLE']], axis=1)

frame['record']=1


###############################################################################
### Pivot table to see see how clusters compare to categories
###############################################################################
pivot=pd.pivot_table(frame, values='record', index='TITLE',
                     columns='Cluster', aggfunc=np.sum)
# at k=4 we only see two categories: one basically machine learning, the other
# all other titles
# at k=5 we see 3 categories: machine learning, one with 
# data engineer and with 1/3 of data science / analyst  

###############################################################################
### Top Terms per cluster
###############################################################################

print("Top terms per cluster:")
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

terms_dict=[]


#save the terms for each cluster and document to dictionaries.  To be used later
#for plotting output.

#dictionary to store terms and titles
cluster_terms={}
cluster_title={}


for i in range(k):
    print("Cluster %d:" % i),
    temp_terms=[]
    temp_titles=[]
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
        terms_dict.append(terms[ind])
        temp_terms.append(terms[ind])
    cluster_terms[i]=temp_terms
    
    print("Cluster %d IDs:" % i, end='')
    temp=frame[frame['Cluster']==i]
    for title in temp['ID']:
        print(' %s,' % title, end='')
        temp_titles.append(title)
    cluster_title[i]=temp_titles

