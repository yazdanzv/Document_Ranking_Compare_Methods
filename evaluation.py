from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluation:
    def __init__(self, pred, true):
        self.pred = pred
        self.true = true

    def accuracy(self):
        return accuracy_score(self.true, self.pred)

    def precision(self):
        return precision_score(self.true, self.pred, average='weighted')

    def recall(self):
        return recall_score(self.true, self.pred, average='weighted')

    def f1score(self):
        return f1_score(self.true, self.pred, average='weighted')

    def evaluate(self):
        results = {"accuracy": self.accuracy(), "precision": self.precision(), "recall": self.recall(),
                   "f1score": self.f1score()}
        return results

    def evaluate_vector_space_model(self):
        pass
