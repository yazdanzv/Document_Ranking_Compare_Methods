import math
from collections import Counter


class Okapi_BM25:
    def __init__(self, doc_tokens: dict):
        self.doc_tokens = doc_tokens
        self.doc_list = [" ".join(value['text']) for _, value in doc_tokens.items()]

    def compute_idf(self, doc_list):  # Compute TFIDF scores and returns them
        idf_scores = {}
        N = len(doc_list)
        all_terms = set([term for doc in doc_list for term in doc.split()])

        for term in all_terms:
            containing_docs = sum([term in doc for doc in doc_list])
            idf_scores[term] = math.log((N - containing_docs + 0.5) / (containing_docs + 0.5) + 1)

        return idf_scores

    def bm25(self, doc, query, idf_scores, k1=1.5, b=0.75):  # BM25 algorithm implemented with given parameters
        doc_terms = Counter(doc.split())
        avgdl = sum([len(doc.split()) for doc in self.doc_list]) / len(self.doc_list)
        score = 0

        for term in query.split():
            if term in idf_scores:
                df = doc_terms[term]
                score += idf_scores[term] * df * (k1 + 1) / (df + k1 * (1 - b + b * len(doc.split()) / avgdl))

        return score

    def rank_documents(self, query, doc_list, top_k):  # Method to rank documents based on the length of the query
        idf_scores = self.compute_idf(doc_list)
        query_length = len(query.split())

        # Adjust BM25 parameters based on query length
        if query_length > 10:  # Long query
            k1 = 2.0
            b = 0.5
        else:  # Short query
            k1 = 1.5
            b = 0.75

        scores = {key: self.bm25(" ".join(self.doc_tokens[key]['text']), query, idf_scores, k1, b) for key, _ in self.doc_tokens.items()}
        ranked_docs = sorted(scores, key=lambda doc: scores[doc], reverse=True)
        ranked_docs = [(ranked_docs[i], scores[ranked_docs[i]]) for i in range(len(ranked_docs))]

        return ranked_docs[:top_k]

    def start(self, query: str, top_k: int):  # Start method to start whole procedure
        ranked_docs = self.rank_documents(query, self.doc_list, top_k)
        return ranked_docs