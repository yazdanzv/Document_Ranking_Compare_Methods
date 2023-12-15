


class Evaluation:
    def __init__(self, top_k_results: list, relevant_docs: list):
        self.top_k_results = top_k_results
        self.relevant_docs = relevant_docs
        print(self.top_k_results)
        print(self.relevant_docs)

    def precision_at_k(self, k: int):
        count = 0
        for i in range(k):
            if self.pred[i]:
                pass

    def evaluate(self):
        results = {"accuracy": self.accuracy(), "precision": self.precision(), "recall": self.recall(),
                   "f1score": self.f1score()}
        return results

    def evaluate_vector_space_model(self):
        pass
