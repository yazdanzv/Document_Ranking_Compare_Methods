import copy


class LanguageModel:
    def __init__(self, docs_tokens: dict):
        self.docs_tokens = docs_tokens
        self.priority = {'trigram': 10, 'bigram': 5, 'unigram': 1}

    def trigram(self, query: str):
        query_tokens = query.split(" ")
        trigram_tokens = [str(query_tokens[i] + " " + query_tokens[i + 1] + " " + query_tokens[i + 2]).strip() for i in
                          range(len(query_tokens) - 2)]
        trigram_results = {key: {k: 0 for k in trigram_tokens} for key, _ in self.docs_tokens.items()}
        for key, value in self.docs_tokens.items():
            for i in range(len(trigram_tokens)):
                if trigram_tokens[i] in " ".join(value['text']):
                    trigram_results[key][trigram_tokens[i]] += 1
        return self.dirichlet_smoothing(trigram_results)

    def bigram(self, query: str):
        query_tokens = query.split(" ")
        bigram_tokens = [str(query_tokens[i] + " " + query_tokens[i + 1]).strip() for i in
                         range(len(query_tokens) - 1)]
        bigram_results = {key: {k: 0 for k in bigram_tokens} for key, _ in self.docs_tokens.items()}
        for key, value in self.docs_tokens.items():
            for i in range(len(bigram_tokens)):
                if bigram_tokens[i] in " ".join(value['text']):
                    bigram_results[key][bigram_tokens[i]] += 1
        return self.dirichlet_smoothing(bigram_results)

    def unigram(self, query: str):
        query_tokens = query.split(" ")
        unigram_tokens = [str(query_tokens[i]).strip() for i in range(len(query_tokens))]
        unigram_results = {key: {k: 0 for k in unigram_tokens} for key, _ in self.docs_tokens.items()}
        for key, value in self.docs_tokens.items():
            for i in range(len(unigram_tokens)):
                if unigram_tokens[i] in " ".join(value['text']):
                    unigram_results[key][unigram_tokens[i]] += 1
        return self.dirichlet_smoothing(unigram_results)

    def dirichlet_smoothing(self, k_gram_result: dict, mu: int = 2000):
        for key1, value1 in k_gram_result.items():
            for key2, value2 in k_gram_result[key1].items():
                total_num_word = 0
                for key3, value3 in self.docs_tokens.items():
                    if key2 in value3['text']:
                        total_num_word += value3['text'].count(key2)
                temp = (value2 + mu * (total_num_word / sum(
                    len(self.docs_tokens[key]['text']) for key, _ in self.docs_tokens.items()))) / (
                               len(self.docs_tokens[key1]['text']) + mu)
                k_gram_result[key1][key2] = copy.deepcopy(temp)
        return k_gram_result

    def score(self, k_gram_results: dict, doc_id: str, status: str):
        return sum(value for _, value in k_gram_results[doc_id].items()) * self.priority[status]

    def language_model_heavy(self, query: str, top_k: int):
        trigram_results = self.trigram(query)
        bigram_results = self.bigram(query)
        unigram_results = self.unigram(query)
        documents_scores = {key: self.score(trigram_results, key, 'trigram') + self.score(bigram_results, key, 'bigram')
                            + self.score(unigram_results, key, 'unigram') for key, _ in self.docs_tokens.items()}
        documents_scores_sorted = sorted(documents_scores, key=lambda doc: documents_scores[doc], reverse=True)
        documents_scores = [(documents_scores_sorted[i], documents_scores[documents_scores_sorted[i]]) for i in range(len(documents_scores_sorted))]

        documents_scores_top_k = documents_scores[:top_k]
        return documents_scores_top_k

    def language_model_light(self, query: str, top_k: int):
        unigram_results = self.unigram(query)
        documents_scores = {key: self.score(unigram_results, key, 'unigram') for key, _ in self.docs_tokens.items()}
        documents_scores_sorted = sorted(documents_scores, key=lambda doc: documents_scores[doc], reverse=True)
        documents_scores = [(documents_scores_sorted[i], documents_scores[documents_scores_sorted[i]]) for i in
                            range(len(documents_scores_sorted))]

        documents_scores_top_k = documents_scores[:top_k]
        return documents_scores_top_k

    def start(self, query: str, top_k: int, status: str = 'light'):
        if status == 'light':
            return self.language_model_light(query, top_k)
        elif status == 'heavy':
            return self.language_model_heavy(query, top_k)


