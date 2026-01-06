from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vectors = np.load(os.path.join(BASE_DIR, "artifacts/vectors.npy"))

def similarity():
    similarity = cosine_similarity(vectors)
    with open(os.path.join(BASE_DIR, "artifacts/similarity.pkl"), "wb") as f:
        pickle.dump(similarity, f)

if __name__ == "__main__":
    similarity()