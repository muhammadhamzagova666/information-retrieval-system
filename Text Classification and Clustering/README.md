# Text Classification and Clustering Application

This application provides a graphical user interface (GUI) for text classification and clustering. It allows users to search for research papers, classify the text, and cluster the documents.

## Directory Structure

.
├── Dictionary.txt
├── Gold Query-Set.txt
├── IDF.json
├── main.py
├── ResearchPapers/
│   ├── 1.txt
│   ├── 2.txt
│   ├── 3.txt
│   ├── ...
├── Stopword-List.txt
├── TFIDF_Matrix.json
└── Vocabulary.json

## Key Files

- [``main.py``]: This is the main entry point for the application. It contains the GUI class and other helper classes like `TextPreprocessor`, `ResearchPapersLoader`, `CustomTfidfVectorizer`, and `VectorSpaceModel`.
- [``Dictionary.txt``]: This file contains a dictionary of words used by the `TextPreprocessor` class.
- [``Stopword-List.txt``]: This file contains a list of stopwords used by the `TextPreprocessor` class.
- [``ResearchPapers/``]: This directory contains the research papers that the application classifies and clusters.
- [``TFIDF_Matrix.json``], [``Vocabulary.json``], [``IDF.json``]: These files are used by the `VectorSpaceModel` class to store the TF-IDF matrix, vocabulary, and IDF values.

## Key Classes

- [`GUI`]: This class provides the graphical user interface for the application. It allows users to search for research papers, classify the text, and cluster the documents.
- [`TextPreprocessor`]: This class is responsible for preprocessing text data. It includes methods for removing punctuation and numbers, tokenizing and stemming text, and loading or creating a dictionary of words.
- [`ResearchPapersLoader`]: This class is responsible for loading research papers from a given directory.
- [`CustomTfidfVectorizer`]: This class is a custom implementation of TF-IDF vectorizer. It includes methods for fitting the model to a corpus and transforming a corpus into TF-IDF vectors.
- [`VectorSpaceModel`]: This class represents a vector space model built on a given corpus. It includes methods for loading or computing the TF-IDF matrix, computing cosine similarity, and processing queries.

## How to Run

To run this application, execute the [`[`main.py`]`] script:

```sh
python main.py
```

## Dependencies

This application requires the following Python libraries:

- PyQt5
- nltk
- sklearn
- numpy

You can install these dependencies using pip:

```sh
pip install PyQt5 nltk sklearn numpy
```

## Contributing

Contributions are welcome. Please submit a pull request with your changes.

## License

This project is licensed under the MIT License.
