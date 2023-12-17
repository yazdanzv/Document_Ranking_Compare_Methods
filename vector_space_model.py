import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorSpaceModel:
    def __init__(self, docs_tokens: dict):
        self.docs_tokens = docs_tokens
        self.docs_vectors = dict()
        self.vector_model = None

    def vectorize_docs(self):
        self.vector_model = TfidfVectorizer(stop_words='english')
        tfidf_docs = self.vector_model.fit_transform([" ".join(value['text']) for _, value in self.docs_tokens.items()])
        return tfidf_docs

    def query_the_docs(self, query: str, top_k: int):
        query_list = [query]
        tfidf_docs = self.vectorize_docs()
        tfidf_query = self.vector_model.transform(query_list)

        # Cosine similarity
        cosine_sim_list = cosine_similarity(tfidf_query, tfidf_docs).flatten()

        documents_scores = {str(i+1): cosine_sim_list[i] for i in range(len(cosine_sim_list))}

        max_heap = [(-similarity, index) for index, similarity in enumerate(cosine_sim_list)]
        heapq.heapify(max_heap)

        top_k_indices = [str(int(heapq.heappop(max_heap)[1]) + 1) for _ in range(top_k)]

        answer = [(doc_id, documents_scores[doc_id]) for doc_id in top_k_indices]
        return answer

