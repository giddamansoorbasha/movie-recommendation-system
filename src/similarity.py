import os
import pickle
import numpy as np
from src.vectorization import vectorize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
VECTORS_PATH = os.path.join(ARTIFACTS_DIR, "vectors.npy")
SIM_PATH = os.path.join(ARTIFACTS_DIR, "similarity.pkl")

def similarity():
    if not os.path.exists(VECTORS_PATH):
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        vectors = vectorize()
    else:
        vectors = np.load(VECTORS_PATH)

    sim = np.load(SIM_PATH, allow_pickle=True) if os.path.exists(SIM_PATH) else None
    if sim is None:
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(vectors)
        with open(SIM_PATH, "wb") as f:
            pickle.dump(sim, f)
    return sim

# run only if file executed directly
if __name__ == "__main__":
    similarity()