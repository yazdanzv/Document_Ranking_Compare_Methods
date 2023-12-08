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
        for query in root_queries.findall('top'):  # Load queries data
            num = query.find('num').text
            title = query.find('title').text
            queries[copy.deepcopy(num)] = {'title': copy.deepcopy(title)}
        self.queries = copy.deepcopy(queries)

        # Load results for evaluation
        results = dict()  # To store data
        with open(self.file_path_results, 'r') as f:
            temp = f.readline()
            print(type(temp))
            while temp != '':
                temp = temp.split(' ')  # Convert it to list
                query_id = copy.deepcopy(temp[0])  # ID of the query
                iteration = copy.deepcopy(temp[1])  # Number of iteration (always equal to 0)
                doc_id = copy.deepcopy(temp[2])  # Document ID
                relevancy = copy.deepcopy(temp[3])[:-1]  # Result of relevancy (0 for not relevant, 1 for relevant)
                if query_id not in results:  # Build results dictionary to store data properly
                    results[query_id] = [{'docno': doc_id, 'relevancy': relevancy}]
                elif query_id in results:
                    results[query_id].append({'docno': doc_id, 'relevancy': relevancy})
                else:
                    raise Exception("ERROR")
                temp = f.readline()  # Read again
            self.results = copy.deepcopy(results)

        @staticmethod
        def case_folding(word: str):
            new_word = word.lower()
            return new_word

        @staticmethod
        def special_characters_remover(word: str):  # Eliminates all the special characters like {, . : ; }
            normalized_word = re.sub(r'[^\w\s]', '', word)
            return normalized_word

        @staticmethod
        def tokenizer(text: str):
            tokens = word_tokenize(text)
            return tokens

        @staticmethod
        def stop_word_remover(tokens: list):
            stop_words = set(stopwords.words('english'))
            new_tokens = []
            for token in tokens:
                if token not in stop_words:
                    new_tokens.append(token)
            return copy.deepcopy(new_tokens)

        @staticmethod
        def stemmer(tokens: list):
            stemmer_obj = PorterStemmer()
            stemmed_tokens = [stemmer_obj.stem(token) for token in tokens]
            return stemmed_tokens

        @staticmethod
        def lemmatizer(tokens: list):
            lemmatizer_obj = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer_obj.lemmatize(token) for token in tokens]
            return lemmatized_tokens


a = Preprocess().load_data()
