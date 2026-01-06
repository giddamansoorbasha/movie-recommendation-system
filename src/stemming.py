from nltk.stem.porter import PorterStemmer
import re

ps = PorterStemmer()

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def stem(text: str):
    if not isinstance(text, str) or not text.strip():
        return ""
    tokens = simple_tokenize(text)
    stemmed = [ps.stem(token) for token in tokens]
    return " ".join(stemmed)