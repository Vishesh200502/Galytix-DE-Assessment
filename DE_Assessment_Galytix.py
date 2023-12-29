#!/usr/bin/env python
# coding: utf-8

# In[44]:


# Importing all the required libraries
import gensim
from gensim.models import KeyedVectors
import pandas as pd

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
import string

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

import numpy as np


# In[21]:


# Load Word2Vec model
location = "GoogleNews-vectors-negative300.bin"
wv = KeyedVectors.load_word2vec_format(location, binary=True, limit=1000000)
wv.save_word2vec_format('vectors.csv')


# In[25]:


wv = KeyedVectors.load_word2vec_format('vectors.csv', binary=False)


# In[22]:


# Load phrases from CSV
phrases_csv_path = "phrases.csv"
phrases_df = pd.read_csv(phrases_csv_path,encoding='latin1')


# In[98]:


# Function to get the word embeddings for a phrase
def get_phrase_embeddings(phrase, model):
    tokens = phrase.split()
    valid_tokens = [token for token in tokens if token in model.key_to_index]
    if not valid_tokens:
        return None
    # Calculate the normalized sum of word embeddings
    phrase_vector = sum(model.get_vector(token) / model.get_vecattr(token, 'count') for token in valid_tokens)
    return phrase_vector


# In[99]:


# Assign embeddings to each word in each phrase
phrases_df['embeddings'] = phrases_df['Phrases'].apply(lambda x: get_phrase_embeddings(x, wv))


# In[100]:


# Function to calculate distances between phrases
def calculate_distances(phrases_df, distance_metric='euclidean'):
    embeddings = np.stack(phrases_df['embeddings'].dropna().to_numpy())

    if distance_metric == 'euclidean':
        distances = euclidean_distances(embeddings, embeddings)
    elif distance_metric == 'cosine':
        distances = cosine_distances(embeddings, embeddings)
    else:
        raise ValueError("Invalid distance metric. Use 'euclidean' or 'cosine'.")
    
    return distances


# In[101]:


# Example usage
euclidean_distances_matrix = calculate_distances(phrases_df, distance_metric='euclidean')
cosine_distances_matrix = calculate_distances(phrases_df, distance_metric='cosine')


# In[102]:


def find_closest_match(user_input, phrases_df, model, distance_metric='euclidean'):
    user_input_embedding = get_phrase_embeddings(user_input, model)
    if user_input_embedding is None:
        return None, None

    # Calculate distances and store them in a new 'distance' column
    phrases_df['distance'] = phrases_df['embeddings'].apply(
        lambda x: euclidean_distances([user_input_embedding], [x]) if distance_metric == 'euclidean'
        else cosine_distances([user_input_embedding], [x])
    )

    # Find the index of the minimum distance using np.argmin
    closest_match_index = np.argmin(phrases_df['distance'].apply(lambda x: x[0]).to_numpy())

    # Retrieve the closest match and distance
    closest_match_phrase = phrases_df.loc[closest_match_index, 'Phrases']
    distance_to_closest_match = phrases_df.loc[closest_match_index, 'distance'][0]

    return closest_match_phrase, distance_to_closest_match


# In[105]:


# Example usage
user_input_phrase = input("Please enter the sentence: ")
closest_match, distance_to_closest_match = find_closest_match(user_input_phrase, phrases_df, wv, distance_metric='cosine')
print(f"Closest Match: {closest_match}, Distance: {distance_to_closest_match}")


# In[ ]:




