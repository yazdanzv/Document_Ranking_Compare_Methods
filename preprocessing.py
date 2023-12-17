import xml.etree.ElementTree as ET
import copy
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In case the nltk library got you with error like ..... package not found use download method for your desired package, all packages that are needed were included here
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

class Preprocess:
    def __init__(self):
        self.file_path_docs = '.\\cranfield-trec-dataset-main\\cran.all.1400.xml'  # XML
        self.file_path_queries = '.\\cranfield-trec-dataset-main\\cran.qry.xml'  # XML
        self.file_path_results = '.\\cranfield-trec-dataset-main\\cranqrel.trec.txt'  # Text file
        self.docs = dict()
        self.queries = dict()
        self.results = dict()
        self.docs_tokens = dict()
        self.queries_tokens = dict()


    def load_data(self):
        # Load documents
        tree_docs = ET.parse(self.file_path_docs)
        root_docs = tree_docs.getroot()
        docs = dict()  # To store data
        for doc in root_docs.findall('doc'):  # Load documents data
            doc_id = doc.find('docno').text
            title = doc.find('title').text
            author = doc.find('author').text
            bib = doc.find('bib').text
            text = doc.find('text').text
            docs[copy.deepcopy(doc_id)] = {'title': copy.deepcopy(title), 'author': copy.deepcopy(author),
                                           'bib': copy.deepcopy(bib),
                                           'text': copy.deepcopy(text)}
        self.docs = copy.deepcopy(docs)

        # Load queries
        tree_queries = ET.parse(self.file_path_queries)
        root_queries = tree_queries.getroot()
        queries = dict()  # To store data
        query_id = 1
        for query in root_queries.findall('top'):  # Load queries data
            # num = query.find('num').text.strip()
            title = query.find('title').text
            queries[copy.deepcopy(str(query_id))] = copy.deepcopy(title)
            query_id += 1
        self.queries = copy.deepcopy(queries)

        # Load results for evaluation
        results = dict()  # To store data
        with open(self.file_path_results, 'r') as f:
            temp = f.readline()
            while temp != '':
                temp = temp.split(' ')  # Convert it to list
                query_id = copy.deepcopy(temp[0])  # ID of the query
                iteration = copy.deepcopy(temp[1])  # Number of iteration (always equal to 0)
                doc_id = copy.deepcopy(temp[2])  # Document ID
                relevancy = copy.deepcopy(temp[3])[:-1]  # Result of relevancy (0 for not relevant, 1 for relevant)
                if query_id not in results:  # Build results dictionary to store data properly
                    results[query_id] = [(doc_id, relevancy)]
                elif query_id in results:
                    results[query_id].append((doc_id, relevancy))
                else:
                    raise Exception("ERROR")
                temp = f.readline()  # Read again
            self.results = copy.deepcopy(results)

    @staticmethod
    def case_folding(text: str):  # Handle upper case characters
        new_word = text.lower()
        return new_word

    @staticmethod
    def special_characters_remover(text: str):  # Eliminates all the special characters like {, . : ; }
        normalized_word = re.sub(r'[^\w\s]', '', text)
        return normalized_word

    @staticmethod
    def tokenizer(text: str):  # Tokenize the text
        tokens = word_tokenize(text)
        return tokens

    @staticmethod
    def stop_word_remover(tokens: list):  # Eliminate stop words
        stop_words = set(stopwords.words('english'))
        new_tokens = []
        for token in tokens:
            if token not in stop_words:
                new_tokens.append(token)
        return copy.deepcopy(new_tokens)

    @staticmethod
    def stemmer(tokens: list):  # Stemming the tokens
        stemmer_obj = PorterStemmer()
        stemmed_tokens = [stemmer_obj.stem(token) for token in tokens]
        return stemmed_tokens

    @staticmethod
    def lemmatizer(tokens: list):  # Lemmatize the tokens
        lemmatizer_obj = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer_obj.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def handle_missing_values(self):
        # Handle missing values in documents
        docs_pop_keys = []
        for key, value in self.docs.items():
            if value['title'] is None or value['author'] is None or value['bib'] is None or value['text'] is None:
                docs_pop_keys.append(key)
        for key in docs_pop_keys:
            self.docs.pop(key, None)

        # Handle missing values in queries
        queries_pop_keys = []
        for key, value in self.queries.items():
            if value is None:
                queries_pop_keys.append(key)
        for key in queries_pop_keys:
            self.queries.pop(key, None)

    def process_data(self):
        # Handle missing values of dataset
        self.handle_missing_values()

        # Process documents
        for key, value in self.docs.items():
            # Process title
            new_title = self.docs[key]['title']  # Load title
            new_title = self.case_folding(new_title)  # Handle upper case characters
            new_title = self.special_characters_remover(new_title)  # Eliminate special characters
            new_title = self.tokenizer(new_title)  # Tokenize title
            new_title = self.stop_word_remover(new_title)  # Eliminate stop words
            new_title = self.stemmer(new_title)  # Apply stemming
            new_title = self.lemmatizer(new_title)  # Apply lemmatization

            # Process text
            new_text = self.docs[key]['text']  # Load title
            new_text = self.case_folding(new_text)  # Handle upper case characters
            new_text = self.special_characters_remover(new_text)  # Eliminate special characters
            new_text = self.tokenizer(new_text)  # Tokenize title
            new_text = self.stop_word_remover(new_text)  # Eliminate stop words
            new_text = self.stemmer(new_text)  # Apply stemming
            new_text = self.lemmatizer(new_text)  # Apply lemmatization

            # Load other data
            author = self.docs[key]['author']
            bib = self.docs[key]['bib']

            # Build document tokens
            self.docs_tokens[key] = {'title': copy.deepcopy(new_title), 'author': copy.deepcopy(author), 'bib': copy.deepcopy(bib), 'text': copy.deepcopy(new_text)}

        # Process queries
        for key, value in self.queries.items():
            new_title = self.queries[key]
            new_title = self.case_folding(new_title)
            new_title = self.special_characters_remover(new_title)
            new_title = self.tokenizer(new_title)
            new_title = self.stop_word_remover(new_title)
            new_title = self.stemmer(new_title)
            new_title = self.lemmatizer(new_title)

            # Build queries tokens
            self.queries_tokens[key] = copy.deepcopy(new_title)


