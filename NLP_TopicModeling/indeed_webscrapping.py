import requests
#from WebFocusedCrawl.items import WebfocusedcrawlItem  # item class
from bs4 import BeautifulSoup 
import re  # regular expressions used to parse tags
import urllib3
import json
import time
import nltk  # used to remove stopwords from tags
from nltk.corpus import stopwords
import string
from collections import Counter

search_urls = ['https://www.indeed.com/jobs?q=title%3A(Data+Scientist)','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=10',
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=20', 'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=30',
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=40','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=50',
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=60', 'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=70',
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=80','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=90', 
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=100','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=110', 
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=120','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=130', 
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=140','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=150', 
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=160','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=170', 
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=180','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=190',
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=200']

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

############# Cleaning data ############
# Load stop words and add new stop words
stop_words = stopwords.words('english')
newStopWords = ['abilities','Abilities', 'ability', 'Ability','able', 'Able', 
                'employer', 'opportunity', 'race', 'color', 'orientation', 
                'gender', 'sex', 'sexual']
stop_words.extend(newStopWords)

def clean_doc(doc):
# split into tokens by white space
    tokens = []
    titles = []
    for d in doc:
        titles = d['BODY']   
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

tokens_BODY = clean_doc(data)

flattened_list = []
for x in tokens_BODY:
    for y in x:
        flattened_list.append(y)
      
token_strings = flattened_list

# Reduce to string and split words
tokens=[]
for i in tokens_BODY:
    tokens.extend(i[0].split())
    
# most common words
Counter(tokens1).most_common(10)

################## Determine words to search on Indeed ############

# return bigrams
bgs = nltk.bigrams(tokens)
#compute frequency distribution for all the bigrams in the text
fdist = nltk.FreqDist(bgs)
print(fdist.most_common(10))
 
# return trigrams
tgs = nltk.trigrams(tokens)
fdist_tg = nltk.FreqDist(tgs)
print(fdist_tg.most_common(10))

fdist.most_common(10)
fdist_tg.most_common(10)


# TF-IDF
tokens1 = [' '.join(word for word in tokens)]

#https://stackoverflow.com/questions/25217510/how-to-see-top-n-entries-of-term-document-matrix-after-tfidf-in-scikit-learn/25219535
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(token_strings)
features_by_gram = defaultdict(list)
for f, w in zip(vectorizer.get_feature_names(), vectorizer.idf_):
    features_by_gram[len(f.split(' '))].append((f, w))
top_n = 2
for gram, features in features_by_gram.items():
    top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
    top_features = [f[0] for f in top_features]
    print ('{}-gram top:'.format(gram), top_features)
    

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
    
with open('jobopenings.json') as f:
    data = json.load(f)

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

######## Conduct new search using identified job titles

# Initial Urls to begin scrapping. 
search_urls_US = ['https://www.indeed.com/jobs?q=title%3A(Data+Scientist)','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=10',
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=20', 'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=30',
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=40','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=50',
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=60', 'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=70',
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=80','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=90', 
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=100','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=110', 
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=120','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=130', 
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=140','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=150', 
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=160','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=170', 
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=180','https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=190',
               'https://www.indeed.com/jobs?q=title%3A%28Data+Scientist%29&start=200',
               'https://www.indeed.com/jobs?q=title%3A(Machine+Learning+Engineer)','https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=10',
               'https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=20', 'https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=30',
               'https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=40','https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=50',
               'https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=60', 'https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=70',
               'https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=80','https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=90', 
               'https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=100','https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=110', 
               'https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=120','https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=130', 
               'https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=140','https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=150', 
               'https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=160','https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=170', 
               'https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=180','https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=190',
               'https://www.indeed.com/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=200',
               'https://www.indeed.com/jobs?q=title%3A(data+engineer)','https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=10',
               'https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=20', 'https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=30',
               'https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=40','https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=50',
               'https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=60', 'https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=70',
               'https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=80','https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=90', 
               'https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=100','https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=110', 
               'https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=120','https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=130', 
               'https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=140','https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=150', 
               'https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=160','https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=170', 
               'https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=180','https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=190',
               'https://www.indeed.com/jobs?q=title%3A%28data+engineer%29&start=200',
               'https://www.indeed.com/jobs?q=title%3A(data+analyst)','https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=10',
               'https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=20', 'https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=30',
               'https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=40','https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=50',
               'https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=60', 'https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=70',
               'https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=80','https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=90', 
               'https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=100','https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=110', 
               'https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=120','https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=130', 
               'https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=140','https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=150', 
               'https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=160','https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=170', 
               'https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=180','https://www.indeed.com/jobs?q=title%3A%28data+analyst%29&start=190']

search_urls_UK = ['https://www.indeed.co.uk/jobs?q=title%3A(Data+Scientist)','https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=10',
               'https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=20', 'https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=30',
               'https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=40','https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=50',
               'https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=60', 'https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=70',
               'https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=80','https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=90', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=100','https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=110', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=120','https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=130', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=140','https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=150', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=160','https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=170', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=180','https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=190',
               'https://www.indeed.co.uk/jobs?q=title%3A%28Data+Scientist%29&start=200',
               'https://www.indeed.co.uk/jobs?q=title%3A(Machine+Learning+Engineer)','https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=10',
               'https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=20', 'https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=30',
               'https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=40','https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=50',
               'https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=60', 'https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=70',
               'https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=80','https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=90', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=100','https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=110', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=120','https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=130', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=140','https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=150', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=160','https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=170', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=180','https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=190',
               'https://www.indeed.co.uk/jobs?q=title%3A%28Machine+Learning+Engineer%29&start=200',
               'https://www.indeed.co.uk/jobs?q=title%3A(data+engineer)','https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=10',
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=20', 'https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=30',
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=40','https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=50',
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=60', 'https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=70',
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=80','https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=90', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=100','https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=110', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=120','https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=130', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=140','https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=150', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=160','https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=170', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=180','https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=190',
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+engineer%29&start=200',
               'https://www.indeed.co.uk/jobs?q=title%3A(data+analyst)','https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=10',
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=20', 'https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=30',
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=40','https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=50',
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=60', 'https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=70',
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=80','https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=90', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=100','https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=110', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=120','https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=130', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=140','https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=150', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=160','https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=170', 
               'https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=180','https://www.indeed.co.uk/jobs?q=title%3A%28data+analyst%29&start=190']


# Scrapping urls
urls = []
# If we want to add US
for link in search_urls_US:
    http = urllib3.PoolManager()
    response = http.request('GET', link)
    soup = BeautifulSoup(response.data)
    for link in soup.find_all('div', class_='title'):
        partial_url = link.a.get('href')
        url = 'https://indeed.com' + partial_url
        if url not in urls:
            urls.append(url)

# If we want to add UK           
for link in search_urls_UK:
    http = urllib3.PoolManager()
    response = http.request('GET', link)
    soup = BeautifulSoup(response.data)
    for link in soup.find_all('div', class_='title'):
        partial_url = link.a.get('href')
        url = 'https://www.indeed.co.uk' + partial_url
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

    