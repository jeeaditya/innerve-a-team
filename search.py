
"""#SEARCH USING COSINE SIMILARITY"""

import pandas as pd
import numpy as np
import os
import re
import operator
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

df = pd.read_csv('Datasets/medicine_database_detailed.csv',
                 engine='python', index_col=0)
df1 = df.copy()
df1['text'] = df1['Medicine Name'] + ' ' + df1['Category'] + \
    ' ' + df1['Composition'] + ' ' + df1['Manufacturer']
df1 = df1.dropna()
df1 = df1.reset_index()

df1['cleaned_text'] = df1['text'].apply(
    lambda x: re.sub('(\d+(\.\d+)?)', r' \1 ', str(x)))
df1['cleaned_text'] = df1['cleaned_text'].apply(lambda x: str(x))
df1['cleaned_text'] = df1['cleaned_text'].apply(lambda x: x.lower())
df1['cleaned_text'] = df1['cleaned_text'].apply(
    lambda x: x.translate(str.maketrans('', '', string.punctuation)))


def wordLemmatizer(data):
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    file_clean_k = pd.DataFrame()
    for index, entry in enumerate(data):

        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if len(word) > 1 and word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
                file_clean_k.loc[index, 'Keyword_final'] = str(Final_words)
                file_clean_k.loc[index, 'Keyword_final'] = str(Final_words)
                file_clean_k = file_clean_k.replace(
                    to_replace="\[.", value='', regex=True)
                file_clean_k = file_clean_k.replace(
                    to_replace="'", value='', regex=True)
                file_clean_k = file_clean_k.replace(
                    to_replace=" ", value='', regex=True)
                file_clean_k = file_clean_k.replace(
                    to_replace='\]', value='', regex=True)
    return file_clean_k


medicine = list(df1['cleaned_text'])

# Create Vocabulary
vocabulary = set()
for doc in medicine:
    vocabulary.update(doc.split(' '))
vocabulary = list(vocabulary)
# Intializating the tfIdf model
tfidf = TfidfVectorizer(vocabulary=vocabulary)
# Fit the TfIdf model
tfidf.fit(medicine)
# Transform the TfIdf model
tfidf_tran = tfidf.transform(medicine)


def gen_vector_T(tokens):
    Q = np.zeros((len(vocabulary)))
    x = tfidf.transform(tokens)
    # print(tokens[0].split(','))
    for token in tokens[0].split(','):
        # print(token)
        try:
            ind = vocabulary.index(token)
            Q[ind] = x[0, tfidf.vocabulary_[token]]
        except:
            pass
    return Q


def cosine_sim(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


def cosine_similarity_T(k, query):
    preprocessed_query = re.sub("\W+", " ", query).strip()
    tokens = word_tokenize(str(preprocessed_query))
    q_df = pd.DataFrame(columns=['q_clean'])
    q_df.loc[0, 'q_clean'] = tokens
    q_df['q_clean'] = wordLemmatizer(q_df.q_clean)
    d_cosines = []

    query_vector = gen_vector_T(q_df['q_clean'])
    for d in tfidf_tran.A:
        d_cosines.append(cosine_sim(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]
    # print("")
    d_cosines.sort()
    a = pd.DataFrame()
    for i, index in enumerate(out):
        #a.loc[i,'index'] = str(index)
        a.loc[i, 'Medicine'] = df1['Medicine Name'][index]
        a.loc[i, 'Manufacturer'] = df1['Manufacturer'][index]
        a.loc[i, 'Uses'] = df1['Uses'][index]
        a.loc[i, 'Side Effects'] = df1['Side Effects'][index]
    for j, simScore in enumerate(d_cosines[-k:][::-1]):
        a.loc[j, 'Score'] = simScore
    return a
