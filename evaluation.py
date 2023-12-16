
class Evaluation:
    def __init__(self, top_k_results: list, relevant_docs: list):
        self.top_k_results = top_k_results
        self.relevant_docs = relevant_docs
        # print(self.top_k_results)
        # print(self.relevant_docs)

    def precision_at_k(self, k: int):
        pass

    def evaluate(self):
        pass

    def evaluate_vector_space_model(self):
        pass
