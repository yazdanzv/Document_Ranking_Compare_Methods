import copy

from preprocessing import Preprocess
from vector_space_model import VectorSpaceModel
from probabilistic_model import Okapi_BM25
from language_model import LanguageModel
from evaluation import Evaluation

# Preprocessing and making docs, queries and results ready
a = Preprocess()
a.load_data()
a.process_data()

# Vector Space Model
b = VectorSpaceModel(a.docs_tokens)
vm_answers = []
for i in range(1, 11):
    ans = b.query_the_docs(" ".join(a.queries_tokens[str(i)]), 11)
    print(f"VSM results, query id = {i}")
    print(ans)
    vm_answers.append(ans)

# Evaluate Vector Space Model
vm_evaluations = []
for i in range(1, 11):
    e = Evaluation(vm_answers[i - 1], " ".join(a.queries_tokens[str(i)]), a.queries_tokens, a.results)
    e_ans = e.k_points_interpolated_average_precision()
    vm_evaluations.append(copy.deepcopy(e_ans))
    print(f"11 points interpolated average precision, query id = {i}")
    print(e_ans)
map_score_vm = sum(vm_evaluations) / 10
print("MAP of Vector Space Model in first 10 queries")
print(map_score_vm)

# Probabilistic Model
c = Okapi_BM25(a.docs_tokens)
pm_answers = []
for i in range(1, 11):
    c_ans = c.start(" ".join(a.queries_tokens[str(i)]), 11)
    pm_answers.append(c_ans)
    print(f"PM Results, query id = {i}")
    print(c_ans)

# Evaluate Probabilistic Model
pm_evaluations = []
for i in range(1, 11):
    f = Evaluation(pm_answers[i - 1], " ".join(a.queries_tokens[str(i)]), a.queries_tokens, a.results)
    f_ans = f.k_points_interpolated_average_precision()
    pm_evaluations.append(f_ans)
    print(f"11 points interpolated average precision, query id = {i}")
    print(f_ans)
map_score_pm = sum(pm_evaluations) / 10
print("MAP of Probabilistic Model in first 10 queries")
print(map_score_pm)

# Language Model
d = LanguageModel(a.docs_tokens)
lm_answers = []
for i in range(1, 11):
    d_ans = d.start(" ".join(a.queries_tokens[str(i)]), 11)
    lm_answers.append(d_ans)
    print(f"LM Results, query id = {i}")
    print(d_ans)

# Evaluate Language Model (Light Version)
lm_evaluations = []
for i in range(1, 11):
    g = Evaluation(lm_answers[i - 1], " ".join(a.queries_tokens[str(i)]), a.queries_tokens, a.results)
    g_ans = g.k_points_interpolated_average_precision()
    lm_evaluations.append(g_ans)
    print(f"11 points interpolated average precision, query id = {i}")
    print(g_ans)
map_score_lm = sum(lm_evaluations) / 10
print("MAP of Language Model in first 10 queries")
print(map_score_lm)
