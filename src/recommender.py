import pickle
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

movies = pd.read_csv(os.path.join(BASE_DIR, "data/movies.csv"))
similarity = pickle.load(open(os.path.join(BASE_DIR, "artifacts/similarity.pkl"), "rb"))

def recommend(movie, top_k=5):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), key=lambda x:x[1], reverse=True)[1:top_k+1]
    recommended_movies = [(movies.iloc[i[0]]['title'],movies.iloc[i[0]]['poster_path']) for i in movies_list]
    return recommended_movies