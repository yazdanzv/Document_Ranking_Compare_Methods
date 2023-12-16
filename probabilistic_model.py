import math
from collections import Counter


class Okapi_BM25:
    def __init__(self, doc_list: list, doc_tokens: dict, query_tokens: dict, results: list):
        self.doc_list = doc_list
        self.doc_tokens = doc_tokens
        self.query_tokens = query_tokens
        self.results = results

    def compute_idf(self, doc_list):
        idf_scores = {}
        N = len(doc_list)
        all_terms = set([term for doc in doc_list for term in doc.split()])

        for term in all_terms:
            containing_docs = sum([term in doc for doc in doc_list])
            idf_scores[term] = math.log((N - containing_docs + 0.5) / (containing_docs + 0.5) + 1)

        return idf_scores

    def bm25(self, doc, query, idf_scores, k1=1.5, b=0.75):
        doc_terms = Counter(doc.split())
        avgdl = sum([len(doc.split()) for doc in self.doc_list]) / len(self.doc_list)
        score = 0

        for term in query.split():
            if term in idf_scores:
                df = doc_terms[term]
                score += idf_scores[term] * df * (k1 + 1) / (df + k1 * (1 - b + b * len(doc.split()) / avgdl))

        return score

    def rank_documents(self, query, doc_list, top_k):
        idf_scores = self.compute_idf(doc_list)
        query_length = len(query.split())

        # Adjust BM25 parameters based on query length
        if query_length > 5:  # Long query
            k1 = 2.0
            b = 0.5
        else:  # Short query
            k1 = 1.5
            b = 0.75

        scores = {doc: self.bm25(doc, query, idf_scores, k1, b) for doc in doc_list}
        ranked_docs = sorted(doc_list, key=lambda doc: scores[doc], reverse=True)
        ranked_docs = [(ranked_docs[i], scores[ranked_docs[i]]) for i in range(len(ranked_docs))]

        return ranked_docs[:top_k]

    def start(self, query: str, top_k: int):
        ranked_docs = self.rank_documents(query, self.doc_list, top_k)
        ranked_docs_ids = []
        for i in range(len(ranked_docs)):
            for key, value in self.doc_tokens.items():
                if ranked_docs[i][0] == " ".join(self.doc_tokens[key]['title']) + " " + " ".join(
                        self.doc_tokens[key]['text']):
                    ranked_docs_ids.append((key, ranked_docs[i][1]))

        return ranked_docs_ids

