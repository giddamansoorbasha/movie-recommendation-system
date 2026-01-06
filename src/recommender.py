import os
import pickle
import pandas as pd
from src.vectorization import vectorize
from src.similarity import similarity as build_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
SIM_PATH = os.path.join(ARTIFACTS_DIR, "similarity.pkl")

movies = pd.read_csv(os.path.join(BASE_DIR, "data/movies.csv"))

if not os.path.exists(SIM_PATH):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    vectorize()        
    build_similarity() 
    
with open(SIM_PATH, "rb") as f:
    similarity = pickle.load(f)

def recommend(movie, top_k=5):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:top_k+1]

    return [
        (movies.iloc[i[0]]['title'], movies.iloc[i[0]]['poster_path'])
        for i in movies_list
    ]