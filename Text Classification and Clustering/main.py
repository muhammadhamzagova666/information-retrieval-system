"""
This script provides a GUI for a text classification and clustering application.
It allows users to search for research papers, classify the text, and cluster the documents.
The `GUI` class is the main entry point for the text classification and clustering application. It provides a graphical user interface (GUI) that allows users to search for research papers, classify the text, and cluster the documents.

The `GUI` class has the following key features:

- Initializes the user interface with various layout components, including search field, search button, and tree views for displaying retrieved documents, text classification, text clustering, and evaluation metrics.
- Implements the `search` method, which processes the user's query, performs text classification and clustering, and displays the relevant documents and evaluation metrics.
- Calculates and updates the text classification evaluation metrics (precision, recall, F1-score, and accuracy) based on the sample test data.
- Calculates and updates the text clustering evaluation metrics (purity, silhouette score, and rand index) based on the sample test data.

The `GUI` class relies on several other classes and modules, including `VectorSpaceModel`, `TextPreprocessor`, and `ResearchPapersLoader`, to handle the underlying text processing and information retrieval tasks.
"""

# Import necessary libraries
import time
import os
import re
import string
import json
import sys
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTreeView, QToolTip
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIcon
from PyQt5.QtCore import QPoint
from nltk.stem import PorterStemmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

# Define file paths
STOPWORDS_FILE = Path('Stopword-List.txt')
DICTIONARY_FILE = Path('Dictionary.txt')
TFIDF_MATRIX_FILE = Path('tfidf_matrix.json')
VOCABULARY_FILE = Path('vocabulary.json')
IDF_FILE = Path('idf.json')
ICON_PATH = Path('path_to_your_icon.png')

# Define class labels for classification
class_labels = {
    "Explainable Artificial Intelligence": [1,2,3,7],
    "Heart Failure": [8,9,11],
    "Time Series Forecasting": [12,13,14,15,16],
    "Transformer Model": [17,18,21],
    "Feature Selection": [22,23,24,25,26]
}

# Create a list of class labels for each document
y = []
for class_name, documents in class_labels.items():
    for _ in documents:
        y.append(class_name)

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Initialize KMeans clustering
kmeans = KMeans(n_clusters=5)

def load_file(file_path):
    """
    Function to load a file and return a list of words.
    
    Args:
        file_path (Path): The path of the file to load.
    
    Returns:
        list: A list of words in the file.
    """
    try:
        with file_path.open('r', encoding='latin-1') as file:
            return [word.strip() for word in file.readlines()]
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []

# Define constants
RESEARCH_PAPERS_DIR = 'ResearchPapers'
FILE_COUNT = 30
ALPHA = 0.05

# Load stopwords
STOPWORDS = load_file(STOPWORDS_FILE)

# Sample test data for evaluation
sample_test_data = {
    "Explainable Artificial Intelligence": "Explainable Artificial Intelligence (XAI) refers to AI algorithms and systems that can provide explanations for their decisions and outputs. This field aims to make AI more transparent and understandable to humans. XAI techniques include generating human-readable explanations, highlighting important features in the data, and providing causal reasoning behind the AI's decisions.",
    "Heart Failure": "Heart failure is a chronic condition where the heart is unable to pump blood efficiently to meet the body's needs. It can be caused by various factors such as coronary artery disease, high blood pressure, and infections. Symptoms of heart failure include shortness of breath, fatigue, and swelling in the legs. Treatment options include lifestyle changes, medication, and in severe cases, surgery.",
    "Time Series Forecasting": "Time series forecasting is a technique used to predict future values based on past observations. It is commonly used in finance, weather forecasting, and sales forecasting. Time series data consists of observations recorded at regular time intervals, such as daily stock prices or hourly temperature readings. Techniques used for time series forecasting include ARIMA models, exponential smoothing, and machine learning algorithms.",
    "Transformer Model": "The transformer model is a type of deep learning model that has been widely adopted for natural language processing tasks. It is known for its attention mechanism, which allows it to focus on different parts of the input sequence. The transformer model has enabled significant advancements in tasks such as machine translation, text generation, and sentiment analysis. Variants of the transformer model include BERT, GPT, and T5.",
    "Feature Selection": "Feature selection is the process of selecting a subset of relevant features from a larger set of features to improve the performance of machine learning models. It helps in reducing overfitting and improving model interpretability. Feature selection techniques include filter methods, wrapper methods, and embedded methods. These techniques evaluate the relevance and importance of features based on various criteria such as correlation, mutual information, and model performance."
}

# GUI class for the application
class GUI(QWidget):
    def __init__(self, vector_space_model, research_papers):
        """
        Initialize the GUI.
        
        Args:
            vector_space_model (VectorSpaceModel): Instance of VectorSpaceModel.
            research_papers (dict): Dictionary containing research papers.
        """
        super().__init__()
        self.vector_space_model = vector_space_model
        self.research_papers = research_papers
        self.init_ui()

    def init_ui(self):
        """
        Initialize the user interface.
        """
        # Layout setup
        layout = QVBoxLayout()
        self.setWindowTitle("Vector Space Information Retrieval Model")
        self.setWindowIcon(QIcon(str(ICON_PATH)))

        # Search layout
        search_layout = QHBoxLayout()
        search_label = QLabel("Enter Query:")
        search_label.setStyleSheet("font-size: 14pt;")
        self.search_field = QLineEdit()
        self.search_field.setStyleSheet("font-size: 14pt;")
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_field)

        self.search_button = QPushButton("Search")
        self.search_button.setStyleSheet("font-size: 14pt; padding: 5px; background-color: #4CAF50; color: white;")
        self.search_button.clicked.connect(self.search)
        search_layout.addWidget(self.search_button)

        layout.addLayout(search_layout)

        # Tree views setup
        self.tree_view = QTreeView()
        self.tree_view.setStyleSheet("font-size: 14pt;")
        self.tree_view.setRootIsDecorated(False)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setModel(None)
        self.add_tree_view(layout, "Retrieved Documents", self.tree_view)

        # Additional tree views
        self.classification_tree_view = QTreeView()
        self.classification_tree_view.setStyleSheet("font-size: 14pt;")
        self.classification_tree_view.setRootIsDecorated(False)
        self.classification_tree_view.setAlternatingRowColors(True)
        self.classification_tree_view.setHeaderHidden(True)
        self.classification_tree_view.setModel(None)
        self.add_tree_view(layout, "Text Classification", self.classification_tree_view)

        # Other tree views (Text Clustering, Evaluation Metrics)
        self.cluster_tree_view = QTreeView()
        self.cluster_tree_view.setStyleSheet("font-size: 14pt;")
        self.cluster_tree_view.setRootIsDecorated(False)
        self.cluster_tree_view.setAlternatingRowColors(True)
        self.cluster_tree_view.setHeaderHidden(True)
        self.cluster_tree_view.setModel(None)
        self.add_tree_view(layout, "Text Clustering", self.cluster_tree_view)

        self.metrics_tree_view = QTreeView()
        self.metrics_tree_view.setStyleSheet("font-size: 14pt;")
        self.metrics_tree_view.setRootIsDecorated(False)
        self.metrics_tree_view.setAlternatingRowColors(True)
        self.metrics_tree_view.setHeaderHidden(True)
        self.metrics_tree_view.setModel(None)
        self.add_tree_view(layout, "Text Classification Evaluation Metrics", self.metrics_tree_view)

        self.clustering_metrics_tree_view = QTreeView()
        self.clustering_metrics_tree_view.setStyleSheet("font-size: 14pt;")
        self.clustering_metrics_tree_view.setRootIsDecorated(False)
        self.clustering_metrics_tree_view.setAlternatingRowColors(True)
        self.clustering_metrics_tree_view.setHeaderHidden(True)
        self.clustering_metrics_tree_view.setModel(None)
        self.add_tree_view(layout, "Text Clustering Evaluation Metrics", self.clustering_metrics_tree_view)

        self.setLayout(layout)
        self.model_timing_tree_view = QTreeView()
        self.model_timing_tree_view.setStyleSheet("font-size: 14pt;")
        self.model_timing_tree_view.setRootIsDecorated(False)
        self.model_timing_tree_view.setAlternatingRowColors(True)
        self.model_timing_tree_view.setHeaderHidden(True)
        self.model_timing_tree_view.setModel(None)
        self.add_tree_view(layout, "Model Timing (s)", self.model_timing_tree_view)

    def add_tree_view(self, layout, label_text, tree_view):
        """
        Add tree view to the layout with a label.

        Args:
            layout: Layout to add the tree view and label.
            label_text (str): Text for the label.
            tree_view: Tree view to add to the layout.
        """
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 14pt;")
        layout.addWidget(label)
        layout.addWidget(tree_view)
    
    # Other methods like update_metrics, classify, purity_score, etc.
    def update_metrics(self, precision, recall, f1, accuracy):
        """
        Update the metrics in the GUI.

        Args:
            precision (float): Precision score.
            recall (float): Recall score.
            f1 (float): F1 score.
            accuracy (float): Accuracy score.
        """
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Metrics", "Values"])
        model.appendRow([QStandardItem("Precision"), QStandardItem(str(precision))])
        model.appendRow([QStandardItem("Recall"), QStandardItem(str(recall))])
        model.appendRow([QStandardItem("F1 Score"), QStandardItem(str(f1))])
        model.appendRow([QStandardItem("Accuracy"), QStandardItem(str(accuracy))])
        self.metrics_tree_view.setModel(model)

    def classify(self):
        """
        Classify the query using the KNN model.
        """
        query = self.search_field.text()
        if query!= '':
            query_vector = self.vector_space_model.vectorizer.transform([query])
            prediction = knn.predict(query_vector)
            model = QStandardItemModel()
            model.setHorizontalHeaderLabels(["Classification"])
            model.appendRow([QStandardItem(prediction[0])])
            self.classification_tree_view.setModel(model)

    def purity_score(self, y_true, y_pred):
        """
        Compute the purity score.

        Args:
            y_true (list): List of true labels.
            y_pred (list): List of predicted labels.

        Returns:
            float: Purity score.
        """
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

    def silhouette_score(self, X, labels):
        """
        Compute the silhouette score.

        Args:
            X (array-like): Input data.
            labels (list): List of labels.

        Returns:
            float: Silhouette score.
        """
        return silhouette_score(X, labels, metric='euclidean')

    def rand_index_score(self, y_true, y_pred):
        """
        Compute the adjusted rand index score.

        Args:
            y_true (list): List of true labels.
            y_pred (list): List of predicted labels.

        Returns:
            float: Adjusted rand index score.
        """
        return adjusted_rand_score(y_true, y_pred)

    def update_clustering_metrics(self, purity, silhouette, rand_index):
        """
        Update the clustering metrics in the GUI.

        Args:
            purity (float): Purity score.
            silhouette (float): Silhouette score.
            rand_index (float): Adjusted rand index score.
        """
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Metrics", "Values"])
        model.appendRow([QStandardItem("Purity"), QStandardItem(str(purity))])
        model.appendRow([QStandardItem("Silhouette Score"), QStandardItem(str(silhouette))])
        model.appendRow([QStandardItem("Rand Index"), QStandardItem(str(rand_index))])
        self.clustering_metrics_tree_view.setModel(model)

    def calculate_clustering_metrics(self):
        """
        Calculate and update clustering metrics.
        """
        # Create a list of true labels in the same order as the documents in your dataset
        labels_true = []
        for class_name, documents in class_labels.items():
            for document in documents:
                labels_true.append(class_name)

        labels_pred = kmeans.labels_

        purity = self.purity_score(labels_true, labels_pred)
        silhouette = self.silhouette_score(self.vector_space_model.tfidf_matrix, labels_pred)
        rand_index = self.rand_index_score(labels_true, labels_pred)

        self.update_clustering_metrics(purity, silhouette, rand_index)

    def cluster(self):
        """
        Cluster the query using the KMeans model.
        """
        query = self.search_field.text()
        if query!= '':
            query_vector = self.vector_space_model.vectorizer.transform([query])
            cluster = kmeans.predict(query_vector)
            model = QStandardItemModel()
            model.setHorizontalHeaderLabels(["Cluster"])
            model.appendRow([QStandardItem(str(cluster[0]))])
            self.cluster_tree_view.setModel(model)

    def update_model_timing(self, model_time):
        #Update the model timing in the GUI

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Model Timing (s)"])
        model.appendRow([QStandardItem(str(model_time))])
        self.model_timing_tree_view.setModel(model)

    def search(self):
        """
        This method is responsible for searching the documents based on the query entered by the user.
        It also calculates the evaluation metrics for the search and updates the GUI with the results.
        """
        # Disable the search button to prevent multiple simultaneous searches
        self.search_button.setEnabled(False)
    
        # Get the query from the search field
        query = self.search_field.text()
    
        # Assign the kmeans object to the instance variable
        self.kmeans = kmeans

        start_time = time.time()
    
        # If the query is not empty, perform the search
        if query != '':
            # Classify and cluster the documents
            self.classify()
            self.cluster()
    
            # Process the queries and get the results
            results = self.vector_space_model.process_queries([query])
    
            # Filter out the relevant documents based on the similarity score
            relevant_docs = [(list(self.research_papers.keys())[idx], sim) for idx, sim in enumerate(results[query]) if sim >= ALPHA]
    
            # Sort the relevant documents in descending order of similarity
            relevant_docs.sort(key=lambda x: x[1], reverse=True)
    
            # If there are any relevant documents, update the tree view with the documents
            if relevant_docs:
                model = QStandardItemModel()
                model.setHorizontalHeaderLabels(["Documents"])
                for doc in relevant_docs:
                    item1 = QStandardItem(doc[0])
                    model.appendRow([item1])
                self.tree_view.setModel(model)
            else:
                # If there are no relevant documents, show a tooltip with the message
                self.tree_view.setModel(None)
                QToolTip.showText(self.search_field.mapToGlobal(QPoint(0, 0)), "No results found for the given query.")
        else:
            # If the query is empty, show a tooltip with the message
            self.tree_view.setModel(None)
            QToolTip.showText(self.search_field.mapToGlobal(QPoint(0, 0)), "There is nothing to be searched.")

        end_time = time.time()

        model_time = (end_time - start_time) 

        self.update_model_timing(model_time)

        # Enable the search button after the search is complete
        self.search_button.setEnabled(True)
    
        # Calculate evaluation metrics
        all_test_predictions = []
        for query in sample_test_data.values():
            query_vector = self.vector_space_model.vectorizer.transform([query])
            prediction = knn.predict(query_vector)
            all_test_predictions.append(prediction[0])
    
        # Calculate precision, recall, f1 score, and accuracy
        precision = precision_score(sample_test_labels, all_test_predictions, average='macro', zero_division=0)
        recall = recall_score(sample_test_labels, all_test_predictions, average='macro')
        f1 = f1_score(sample_test_labels, all_test_predictions, average='macro')
        accuracy = accuracy_score(sample_test_labels, all_test_predictions)
    
        # Update the metrics on the GUI
        self.update_metrics(precision, recall, f1, accuracy)
    
        # Calculate and update clustering metrics
        self.calculate_clustering_metrics()
    
    def run(self):
        """
        This method is responsible for running the GUI application.
        """
        # Show the GUI
        self.show()
    
        # Exit the application when the GUI is closed
        sys.exit(app.exec_())

class TextPreprocessor:
    """
    This class is responsible for preprocessing text data. It includes methods for removing punctuation and numbers,
    tokenizing and stemming text, and loading or creating a dictionary of words.
    """
    def __init__(self, corpus: List[str]):
        """
        Initialize the TextPreprocessor with a corpus of text.
        
        Args:
            corpus (List[str]): The corpus of text to preprocess.
        """
        self.corpus = corpus
        self.stemmer = PorterStemmer()
        self.token_pattern = re.compile(r"\w+(?:['-,]\w+)?|[^_\w\s]")
        self.dictionary = {}
        self.dictionary = self.load_or_create_dictionary()

    def remove_punctuation_and_numbers(self, text: str) -> str:
        """
        Remove punctuation and numbers from a given text.
        
        Args:
            text (str): The text to preprocess.
            
        Returns:
            str: The preprocessed text.
        """
        return text.translate(str.maketrans('', '', string.punctuation + string.digits))

    def tokenize_and_stem(self, text: str) -> List[str]:
        """
        Tokenize and stem a given text.
        
        Args:
            text (str): The text to tokenize and stem.
            
        Returns:
            List[str]: The list of stemmed tokens.
        """
        text = self.remove_punctuation_and_numbers(text.lower())
        tokens = self.token_pattern.findall(text)
        return [self.stemmer.stem(token) for token in tokens if token.isalpha() and token not in STOPWORDS]

    def load_or_create_dictionary(self) -> Dict[str, int]:
        """
        Load or create a dictionary of words from the corpus.
        
        Returns:
            Dict[str, int]: The dictionary of words.
        """
        if DICTIONARY_FILE.exists():
            with DICTIONARY_FILE.open('r') as file:
                self.dictionary = json.load(file)
        else:
            for doc in self.corpus:
                tokens = self.tokenize_and_stem(doc)
                for token in tokens:
                    self.dictionary[token] = self.dictionary.get(token, 0) + 1
            with DICTIONARY_FILE.open('w') as file:
                json.dump(self.dictionary, file)
        return self.dictionary

class ResearchPapersLoader:
    """
    This class is responsible for loading research papers from a given directory.
    """
    def __init__(self, directory, file_count):
        """
        Initialize the ResearchPapersLoader with the directory and file count.
        
        Args:
            directory (str): The directory where the research papers are stored.
            file_count (int): The number of files to load.
        """
        self.directory = directory
        self.file_count = file_count

    def generate_file_paths(self):
        """
        Generate the file paths for the research papers.
        
        Returns:
            list: A list of file paths for the research papers.
        """
        # Generate file paths for the given range and check if the file exists
        return [f'{self.directory}/{i}.txt' for i in range(1, self.file_count + 1) if os.path.exists(f'{self.directory}/{i}.txt')]

    def load_research_papers(self):
        """
        Load the research papers from the file paths.
        
        Returns:
            dict: A dictionary where the keys are the file names and the values are the content of the files.
        """
        # Generate the file paths
        file_paths = self.generate_file_paths()
        research_papers = {}
        # For each file path, open the file and read the content
        for path in file_paths:
            with open(path, 'r', encoding='latin-1') as file:
                content = file.read()
                # Add the content to the dictionary with the file name as the key
                research_papers[os.path.basename(path)] = content
        return research_papers

class CustomTfidfVectorizer:
    """
    This class is a custom implementation of TF-IDF vectorizer.
    It includes methods for fitting the model to a corpus and transforming a corpus into TF-IDF vectors.
    """
    def __init__(self, tokenizer):
        """
        Initialize the CustomTfidfVectorizer with a tokenizer.
        
        Args:
            tokenizer (function): The function to use for tokenizing the text.
        """
        self.tokenizer = tokenizer
        self.vocabulary_ = {}  # The vocabulary obtained from the corpus
        self.idf_ = []  # The inverse document frequencies of the words in the vocabulary

    def fit(self, corpus):
        """
        Fit the model to a corpus. This involves building the vocabulary and calculating the IDF values.
        
        Args:
            corpus (list): The corpus to fit the model to.
            
        Returns:
            CustomTfidfVectorizer: The fitted model.
        """
        total_documents = len(corpus)
        document_count = defaultdict(int)
        # For each document in the corpus
        for doc in corpus:
            # Tokenize the document and get the unique words
            words = set(self.tokenizer(doc))
            # For each unique word, increment its document count
            for word in words:
                document_count[word] += 1
        # For each word and its document count
        for word, count in document_count.items():
            # Add the word to the vocabulary and calculate its IDF
            self.vocabulary_[word] = len(self.vocabulary_)
            idf = 1 + np.log(total_documents / (1 + count))
            self.idf_.append(idf)
        return self

    def transform(self, corpus):
        """
        Transform a corpus into TF-IDF vectors.
        
        Args:
            corpus (list): The corpus to transform.
            
        Returns:
            np.array: The TF-IDF vectors.
        """
        matrix = []
        # For each document in the corpus
        for doc in corpus:
            # Tokenize the document and get the word counts
            words = self.tokenizer(doc)
            word_count = Counter(words)
            # Initialize a row for the document
            row = [0] * len(self.vocabulary_)
            # For each word and its count
            for word, count in word_count.items():
                # If the word is in the vocabulary, calculate its TF-IDF and add it to the row
                if word in self.vocabulary_:
                    tf = count / len(words)
                    idf = self.idf_[self.vocabulary_[word]]
                    row[self.vocabulary_[word]] = tf * idf
            # Add the row to the matrix
            matrix.append(row)
        return np.array(matrix)

    def fit_transform(self, corpus):
        """
        Fit the model to a corpus and transform the corpus into TF-IDF vectors.
        
        Args:
            corpus (list): The corpus to fit the model to and transform.
            
        Returns:
            np.array: The TF-IDF vectors.
        """
        self.fit(corpus)
        return self.transform(corpus)

class VectorSpaceModel:
    """
    This class represents a vector space model built on a given corpus.
    It includes methods for loading or computing the TF-IDF matrix, computing cosine similarity, and processing queries.
    """
    def __init__(self, corpus: List[str]):
        """
        Initialize the VectorSpaceModel with a corpus of text.
        
        Args:
            corpus (List[str]): The corpus of text to build the model on.
        """
        self.corpus = corpus
        self.vectorizer = CustomTfidfVectorizer(tokenizer=TextPreprocessor(corpus).tokenize_and_stem)
        self.tfidf_matrix = self.load_or_compute_tfidf_matrix()

    def load_or_compute_tfidf_matrix(self):
        """
        Load the TF-IDF matrix, vocabulary, and IDF values from files if they exist, otherwise compute them.
        
        Returns:
            np.array: The TF-IDF matrix.
        """
        # If the TF-IDF matrix, vocabulary, and IDF values exist in files, load them
        if TFIDF_MATRIX_FILE.exists() and VOCABULARY_FILE.exists() and IDF_FILE.exists():
            with TFIDF_MATRIX_FILE.open('r') as file:
                self.tfidf_matrix = np.array(json.load(file))
            with VOCABULARY_FILE.open('r') as file:
                self.vectorizer.vocabulary_ = json.load(file)
            with IDF_FILE.open('r') as file:
                self.vectorizer.idf_ = json.load(file)
        else:
            # Otherwise, compute the TF-IDF matrix, vocabulary, and IDF values and save them to files
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus).tolist()
            with TFIDF_MATRIX_FILE.open('w') as file:
                json.dump(self.tfidf_matrix, file)
            with VOCABULARY_FILE.open('w') as file:
                json.dump(self.vectorizer.vocabulary_, file)
            with IDF_FILE.open('w') as file:
                json.dump(self.vectorizer.idf_, file)
        return self.tfidf_matrix

    def compute_cosine_similarity(self, query):
        """
        Compute the cosine similarity between a query and each document in the corpus.
        
        Args:
            query (str): The query to compute the cosine similarity for.
            
        Returns:
            np.array: The cosine similarities.
        """
        # Transform the query into a vector
        query_vector = self.vectorizer.transform([query])[0]
        cosine_similarities = []
        # For each document vector in the TF-IDF matrix
        for doc_vector in self.tfidf_matrix:
            # Compute the dot product between the query vector and the document vector
            dot_product = np.dot(query_vector, doc_vector)
            # Compute the norms of the query vector and the document vector
            norm_query = np.linalg.norm(query_vector)
            norm_doc = np.linalg.norm(doc_vector)
            # Compute the cosine similarity and add it to the list
            similarity = dot_product / (norm_query * norm_doc) if norm_query != 0 and norm_doc != 0 else 0
            cosine_similarities.append(similarity)
        return np.array(cosine_similarities)

    def process_queries(self, queries):
        """
        Process a list of queries by computing the cosine similarity for each one.
        
        Args:
            queries (List[str]): The queries to process.
            
        Returns:
            dict: A dictionary where the keys are the queries and the values are the cosine similarities.
        """
        results = {}
        # For each query
        for query in queries:
            # Compute the cosine similarity and add it to the results
            cosine_similarities = self.compute_cosine_similarity(query)
            results[query] = cosine_similarities
        return results

if __name__ == "__main__":
    # Create a QApplication instance
    app = QApplication(sys.argv)

    # Create a ResearchPapersLoader instance and load the research papers
    loader = ResearchPapersLoader(RESEARCH_PAPERS_DIR, FILE_COUNT)
    research_papers = loader.load_research_papers()

    # Create a corpus from the research papers
    corpus = list(research_papers.values())

    # Create a VectorSpaceModel instance from the corpus
    vector_space_model = VectorSpaceModel(corpus)

    # Determine the number of neighbors for the KNN classifier
    n_neighbors = int(np.sqrt(len(y)))
    # Create a KNN classifier and fit it to the TF-IDF matrix
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(vector_space_model.tfidf_matrix, y)

    # Determine the number of clusters for the KMeans clustering
    n_clusters = int(np.sqrt(len(y)))
    # Create a KMeans instance and fit it to the TF-IDF matrix
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(vector_space_model.tfidf_matrix)

    # Create a test corpus and labels from the sample test data
    sample_test_corpus = list(sample_test_data.values())
    sample_test_labels = list(sample_test_data.keys())

    # Transform the test corpus into vectors
    sample_test_vectors = vector_space_model.vectorizer.transform(sample_test_corpus)

    # Predict the labels for the test vectors
    sample_test_predictions = knn.predict(sample_test_vectors)

    # Compute the precision, recall, F1 score, and accuracy of the predictions
    precision = precision_score(sample_test_labels, sample_test_predictions, average='macro', zero_division=0)
    recall = recall_score(sample_test_labels, sample_test_predictions, average='macro')
    f1 = f1_score(sample_test_labels, sample_test_predictions, average='macro')
    accuracy = accuracy_score(sample_test_labels, sample_test_predictions)

    # Create a GUI instance and run it
    gui = GUI(vector_space_model, research_papers)
    gui.run()