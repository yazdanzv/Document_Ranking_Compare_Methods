from preprocessing import Preprocess
from vector_space_model import VectorSpaceModel
from probabilistic_model import Okapi_BM25
from language_model import LanguageModel
from evaluation import Evaluation

a = Preprocess()
a.load_data()
a.process_data()
# print(a.queries_tokens)
# print(a.docs)
# print(a.queries)
# print(a.results)
# print("queries tokens")
# print(a.queries_tokens)

b = VectorSpaceModel(a.docs_tokens)
ans = b.query_the_docs(" ".join(a.queries_tokens['1']), 11)
print("VSM results")
print(ans)

e = Evaluation(ans, a.queries_tokens, a.results)
e_ans = e.k_points_interpolated_average_precision()

# # probabilistic model
# c = Okapi_BM25(a.docs_tokens)
# print("PM Results")
# c_ans = c.start(" ".join(a.queries_tokens['1']), 11)
# print(c_ans)
#
# # Language Model
# d = LanguageModel(a.docs_tokens)
# d_ans = d.start(" ".join(a.queries_tokens['1']), 11)
# print("LM Results")
# print(d_ans)
