import copy
from evaluation import Evaluation
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorSpaceModel:
    def __init__(self, docs_tokens: dict, queries_tokens: dict, results: dict):
        self.docs_tokens = docs_tokens
        self.queries_tokens = queries_tokens
        self.docs = []
        self.queries = []
        self.query = str()
        self.make_docs_ready()
        self.make_queries_ready()
        self.docs_vectors = dict()
        self.queries_vectors = dict()
        self.results = results
        self.tfidf_docs = None
        self.vector_model = None
        self.ranked_list = list()
        self.relevant_docs = list()
        self.top_k_results = list()
        self.precision_recall_list = list()

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
        self.query = query
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
                if ranked_documents[i][0] == " ".join(self.docs_tokens[key]['title']) + " " + " ".join(
                        self.docs_tokens[key]['text']):
                    ranked_list.append((key, ranked_documents[i][1]))

        self.ranked_list = ranked_list  # Sorted
        answer = [ranked_list[i][0] for i in range(top_k)]  # Top k answer
        self.top_k_results = answer
        return answer

    def find_relevant_docs(self):  # Make precision - recall list
        query_id = str()
        for i in range(len(list(self.queries_tokens.keys()))):
            temp = self.query.split(" ")
            if temp == self.queries_tokens[str(i + 1)]:
                query_id = str(i + 1)

        relevant_docs = self.results[query_id]
        self.relevant_docs = relevant_docs

    def evaluate(self):
        eval = Evaluation(self.top_k_results, self.relevant_docs)

