# Movie Recommendation System 🎬

End-to-end content-based movie recommendation system built with
**Python, TF-IDF, Cosine Similarity, FastAPI, and Streamlit**.

## Features
- Data cleaning and preprocessing
- Text vectorization using TF-IDF
- Cosine similarity–based recommendations
- FastAPI backend with REST endpoint
- Streamlit frontend UI
- Poster integration
- Deployed-ready architecture

## Tech Stack
- Python
- Pandas, NumPy, Scikit-learn
- FastAPI
- Streamlit

## Project Structure
- `src/` → core ML logic
- `api/` → FastAPI backend
- `app/` → Streamlit frontend
- `artifacts/` → precomputed model assets

## How to Run Locally
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

## API Endpoint
- GET /recommend/{movie}?k=5
- Returns recommended movies with poster links.