from fastapi import FastAPI
import pickle
import pandas as pd
from src.recommender import recommend

app = FastAPI()

@app.get("/recommend/{movie}")
def get_recommendations(movie: str, k: int = 5):
    try:
        recs = recommend(movie, k)
    except:
        return {"error": "Movie not found"}
    names, posters = zip(*recs)
    return {
        "movie": movie,
        "recommendations": [
            {"title": n, "poster": p}
            for n, p in zip(names, posters)
        ]
    }