# Vector Space Information Retrieval Model

This is a Python application that allows users to search for relevant research papers based on a given query. It uses a custom TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to compute the similarity between the query and the research papers, and displays the most relevant documents in a tree view.

## Features

- **Loading research papers from a directory**: The application can load research papers from a specified directory. Each research paper is a separate text file.

- **Preprocessing the text of the research papers**: The application preprocesses the text of the research papers by removing punctuation, tokenizing the text into individual words, and applying stemming to reduce words to their root form.

- **Creating a custom TF-IDF vectorizer**: The application uses a custom TF-IDF vectorizer to compute the similarity between the query and the research papers. The TF-IDF vectorizer transforms the text of the research papers into a matrix of TF-IDF features.

- **Displaying the search results in a GUI**: The application displays the search results in a graphical user interface (GUI) with a search field for entering queries and a tree view for displaying the search results.

- **Utility functions**: The application includes utility functions to load and save the TF-IDF matrix, vocabulary, and IDF values to disk to avoid recomputing them on each run.

## How it works

The application uses several classes to perform its operations:

- `GUI`: This class is responsible for creating the graphical user interface for the application. It includes a search field for entering queries and a tree view for displaying the search results.

- `TextPreprocessor`: This class is responsible for preprocessing the text of the research papers. It removes punctuation, tokenizes the text, and applies stemming.

- `ResearchPapersLoader`: This class is responsible for loading the research papers from a specified directory.

- `CustomTfidfVectorizer`: This class is responsible for creating a custom TF-IDF vectorizer. It computes the TF-IDF matrix for the corpus of research papers.

- `VectorSpaceModel`: This class is responsible for creating a vector space model from the corpus of research papers. It processes the queries and computes the cosine similarity between the query and the documents.

## How to run

1. Ensure you have Python 3 and PyQt5 installed on your machine.
2. Clone this repository.
3. Navigate to the directory containing the `main.py` file.
4. Run `python main.py` in your terminal.

## Dependencies

- Python 3
- PyQt5
- NLTK
- NumPy

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
