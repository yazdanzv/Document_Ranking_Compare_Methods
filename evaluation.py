from sklearn.metrics import precision_recall_curve
import numpy as np


class Evaluation:
    def __init__(self, top_k_results: list,query: str, query_tokens: dict, results: list):
        self.top_k_results = top_k_results
        self.query_tokens = query_tokens
        self.query = query
        self.query_id = self.find_query_id()
        self.relevant_docs = [doc_id[0] for doc_id in results[self.query_id] if doc_id[1] == '1']

    def find_query_id(self):  # Find the id of given query
        for key, value in self.query_tokens.items():
            if self.query == " ".join(value):
                return key

    def recall_precision_pair(self):  # Find the relevant documents for given query id
        scores = [score[1] for score in self.top_k_results]
        true_labels = [1 if self.top_k_results[i][0] in self.relevant_docs else 0 for i in range(len(self.top_k_results))]

        # Calculate precision and recall for various thresholds
        precision, recall, thresholds = precision_recall_curve(true_labels, scores)

        # Include (1, 0) as an additional point, representing no recall and full precision
        precision = np.append(precision, 0)
        recall = np.append(recall, 1)

        # Store recall-precision pairs
        recall_precision_pairs = list(zip(recall, precision))
        recall_precision_pairs = sorted(recall_precision_pairs, key=lambda x: (x[0], -x[1]))
        return recall_precision_pairs

    def k_points_interpolated_average_precision(self, k_points: int = 11):  # Calculate k points interpolated average precision
        recall_precisions_pairs = self.recall_precision_pair()

        # Determine the recall levels to interpolate
        recall_levels = [i / (k_points - 1) for i in range(k_points)]

        # Calculate max precision for each recall level
        max_precisions = []
        for level in recall_levels:
            max_precision = max((precision for recall, precision in recall_precisions_pairs if recall >= level), default=0)
            max_precisions.append(max_precision)

        # Calculate average precision
        interpolated_avg_precision = sum(max_precisions) / k_points

        return interpolated_avg_precision

