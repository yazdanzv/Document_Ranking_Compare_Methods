from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorSpaceModel:
    def __init__(self, docs_tokens: dict, queries_tokens: dict):
        self.docs_tokens = docs_tokens
        self.queries_tokens = queries_tokens
        self.docs_vectors = dict()
        self.queries_vectors = dict()

    def vectorize(self):
        vectorizer = TfidfVectorizer(stop_words='english')
        # tfidf_docs = vectorizer.fit_transform()
