import copy

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorSpaceModel:
    def __init__(self, docs_tokens: dict, queries_tokens: dict):
        self.docs_tokens = docs_tokens
        self.queries_tokens = queries_tokens
        self.docs = []
        self.queries = []
        self.make_docs_ready()
        self.make_queries_ready()
        self.docs_vectors = dict()
        self.queries_vectors = dict()
        self.tfidf_docs = None
        self.vector_model = None

    def make_docs_ready(self):
        # Policy : joint title, author, bib and text together for each document and make list of all documents
        for key, value in self.docs_tokens.items():
            new_info = " ".join(value['title']) + " " + " ".join(value['text'])
            self.docs.append(copy.deepcopy(new_info))

    def make_queries_ready(self):
        # Policy : Same as documents
        for key, value in self.queries_tokens.items():
            new_info = " ".join(value)
            self.queries.append(copy.deepcopy(new_info))

    def vectorize_docs(self):
        self.vector_model = TfidfVectorizer(stop_words='english')
        self.tfidf_docs = self.vector_model.fit_transform(self.docs)

    def query_the_docs(self, query: str, top_k: int):
        # query = " ".join(query)
        query_list = [query]
        tfidf_query = self.vector_model.transform(query_list)

        # Cosine similarity
        cosine_sim_list = cosine_similarity(tfidf_query, self.tfidf_docs).flatten()

        # Rank docs
        document_rankings = np.argsort(cosine_sim_list)[::-1]

        # Build Ranked docs indexes
        ranked_documents = [(self.docs[index], cosine_sim_list[index]) for index in document_rankings]
        ranked_list = []
        for i in range(len(ranked_documents)):
            for key, value in self.docs_tokens.items():
                if ranked_documents[i][0] == " ".join(self.docs_tokens[key]['title']) + " " + " ".join(self.docs_tokens[key]['text']):
                    ranked_list.append((key, ranked_documents[i][1]))
        # print(ranked_list)

        answer = [ranked_list[i] for i in range(top_k)]
        return answer

