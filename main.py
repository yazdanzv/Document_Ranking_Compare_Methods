from preprocessing import Preprocess
from vector_space_model import VectorSpaceModel

a = Preprocess()
a.load_data()
a.process_data()
# print(a.queries_tokens)
# print(a.docs)
# print(a.queries)
# print(a.results)

b = VectorSpaceModel(a.docs_tokens, a.queries_tokens)
print(b.docs)
b.vectorize_docs()
print(b.tfidf_docs)
ans = b.query_the_docs('what are the existing solutions for hypersonic viscous interactions over an insulated flat plate .', 5)
print(ans)