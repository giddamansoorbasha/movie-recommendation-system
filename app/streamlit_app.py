import streamlit as st
import pandas as pd
import requests
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

movies = pd.read_csv(os.path.join(BASE_DIR, 'data/movies.csv'))


st.title("Movie Recommendation System")
selected_movie = st.selectbox("Select a movie to recommend", movies['title'].values)

if st.button("Recommend"):
    url = f"{API_URL}/recommend/{selected_movie}"
    response = requests.get(url).json()

    if "error" in response:
        st.error(response["error"])
    else:
        recs = response["recommendations"]

        cols1 = st.columns(5)
        for col, rec in zip(cols1, recs[:5]):
            with col:
                st.image(rec["poster"])
                st.caption(rec["title"])

        cols2 = st.columns(5)
        for col, rec in zip(cols2, recs[5:]):
            with col:
                st.image(rec["poster"])
                st.caption(rec["title"])