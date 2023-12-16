from preprocessing import Preprocess
from vector_space_model import VectorSpaceModel
from probabilistic_model import Okapi_BM25

a = Preprocess()
a.load_data()
a.process_data()
# print(a.queries_tokens)
# print(a.docs)
# print(a.queries)
print(a.results)
print("queries tokens")
print(a.queries_tokens)

b = VectorSpaceModel(a.docs_tokens, a.queries_tokens, a.results)
# print(b.docs)
b.vectorize_docs()
# print(b.tfidf_docs)
ans = b.query_the_docs(" ".join(a.queries_tokens['1']), 11)
b.find_relevant_docs()
print("VSM results")
print(ans)
# b.evaluate()

# probabilistic model
c = Okapi_BM25(b.docs, a.docs_tokens, a.queries_tokens, a.results)
print("PM Results")
c_ans = c.start(" ".join(a.queries_tokens['1']), 11)
print(c_ans)
# print(len(c_ans))
# print(type(c_ans))
