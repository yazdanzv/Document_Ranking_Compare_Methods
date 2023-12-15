from preprocessing import Preprocess
from vector_space_model import VectorSpaceModel

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
ans = b.query_the_docs(" ".join(a.queries_tokens['1']), 5)
print(ans)
