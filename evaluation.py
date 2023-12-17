from sklearn.metrics import precision_recall_curve
import numpy as np


class Evaluation:
    def __init__(self, query_tokens: dict, results: list):
        self.query_tokens = query_tokens
        self.results = results

    def find_query_id(self, query: str):
        for key, value in self.query_tokens.items():
            if query == " ".join(value):
                return key

    def find_relevant_docs(self, query_id: str):
        relevant_docs = [doc_id[0] for doc_id in self.results[query_id] if doc_id[1] == '1']
        return relevant_docs

    def recall_precision_pair(self, relevant_docs: list, top_k_results: list):
        scores = [score[1] for score in top_k_results]
        true_labels = [1 if top_k_results[i][0] in relevant_docs else 0 for i in range(len(self.top_k_results))]

        # Calculate precision and recall for various thresholds
        precision, recall, thresholds = precision_recall_curve(true_labels, scores)

        # Include (1, 0) as an additional point, representing no recall and full precision
        precision = np.append(precision, 0)
        recall = np.append(recall, 1)

        # Store recall-precision pairs
        recall_precision_pairs = list(zip(recall, precision))
        recall_precision_pairs = sorted(recall_precision_pairs, key=lambda x: (x[0], -x[1]))
        print(recall_precision_pairs)
        return recall_precision_pairs

    def k_points_interpolated_average_precision(self, relevant_docs: list, top_k_result: list, k_points: int = 11):

        recall_precisions_pairs = self.recall_precision_pair(relevant_docs, top_k_result)

        # Determine the recall levels to interpolate
        recall_levels = [i / (k_points - 1) for i in range(k_points)]

        # Calculate max precision for each recall level
        max_precisions = []
        for level in recall_levels:
            max_precision = max((precision for recall, precision in recall_precisions_pairs if recall >= level),
                                default=0)
            max_precisions.append(max_precision)

        # Calculate average precision
        interpolated_avg_precision = sum(max_precisions) / k_points

        return interpolated_avg_precision

    def mean_average_precision(self, query_list, top_k_list):
        mean_list = []
        average_precision = dict()
        for i in range(len(query_list)):
            query_id = self.find_query_id(query_list[i])
            relevant_docs = self.find_relevant_docs(query_id)
            average_precision[query_id] = self.k_points_interpolated_average_precision(relevant_docs, top_k_list)
