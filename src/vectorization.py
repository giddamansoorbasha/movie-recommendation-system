from sklearn.feature_extraction.text import TfidfVectorizer
from src.stemming import stem
import numpy as np
import pickle
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(BASE_DIR, 'data/movies.csv'))
tfidf  = TfidfVectorizer(max_features=5000, stop_words='english')

def vectorize():
    tags = df['tags'].apply(stem)
    vectors = tfidf .fit_transform(tags).toarray()
    with open(os.path.join(BASE_DIR, "artifacts/vectorizer.npy"), 'wb') as f:
        pickle.dump(tfidf , f)
    np.save(os.path.join(BASE_DIR, "artifacts/vectors.npy"), vectors)
