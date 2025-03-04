# Text-To-Search: Information Retrieval System

> Powerful document indexing and search capabilities with advanced ranking algorithms for precise information retrieval.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

## Overview

Text-To-Search is a comprehensive information retrieval system that processes, indexes, and searches through large document collections with high precision and recall. Built for researchers, developers, and information specialists, it utilizes modern IR algorithms to effectively retrieve and rank relevant information.

### Key Features

- **Multiple Retrieval Models**: 
  - Boolean Retrieval with logical operators (AND, OR, NOT)
  - Vector Space Model with TF-IDF weighting
  - Proximity search capabilities

- **Advanced Indexing**:
  - Inverted index creation and maintenance
  - Positional indexing for phrase queries
  - Efficient document parsing and tokenization

- **Query Processing**:
  - Query expansion with synonyms
  - Spelling correction
  - Stemming and stop word removal

- **User Interface**:
  - Interactive GUI for search operations
  - Results visualization with relevance scores
  - Document preview functionality

### Target Audience

- **Researchers** looking for efficient literature review tools
- **Developers** building search functionality into applications
- **Information Specialists** processing and analyzing large document collections
- **Students** learning about information retrieval systems

### What Makes Text-To-Search Different?

- Simple yet powerful Python implementation for easy customization
- Combination of classical IR techniques with modern algorithms
- Transparent ranking mechanism with explainable relevance scores
- Efficient processing suitable for both small and large document collections

## Technology Stack

- **Programming Language**: Python 3.8+
- **Core Libraries**:
  - NLTK 3.6.2 (Natural Language Toolkit)
  - PyQt5 5.15.4 (GUI framework)
  - NumPy 1.20.3 (Mathematical operations)
  - scikit-learn 0.24.2 (TF-IDF implementation)
- **Data Storage**: JSON for index persistence
- **Development Tools**:
  - Git for version control
  - pytest for testing

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning the repository)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/muhammadhamzagova666/information-retrieval-system.git
   cd information-retrieval-system
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your document collection**:
   - Place your text documents in the `data/documents` directory
   - Alternatively, configure the system to use your custom data source

5. **Build the index**:
   ```bash
   python -m Text-To-Search.indexer --data-dir data/documents --output index.json
   ```

## Usage Guide

### Basic Search Operations

1. **Start the application**:
   ```bash
   python -m Text-To-Search.main
   ```

2. **Boolean Queries**:
   ```
   machine AND learning NOT neural
   information OR retrieval
   ```

3. **Phrase Queries**:
   ```
   "information retrieval system"
   ```

4. **Proximity Queries**:
   ```
   "machine learning" /5 algorithm
   ```

5. **Using the GUI**:

   - Enter your query in the search box
   - Select the retrieval model (Boolean or Vector Space)
   - Adjust relevance parameters if needed
   - Review ranked results with document previews

## Configuration & Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# Application settings
MAX_RESULTS=50
ENABLE_QUERY_EXPANSION=true
DEBUG_MODE=false

# Index settings
STOPWORDS_FILE=data/stopwords.txt

# Performance settings
THREADING_ENABLED=true
MAX_THREADS=4
```

Load these variables in your code using:

```python
from dotenv import load_dotenv
import os

load_dotenv()
max_results = int(os.getenv("MAX_RESULTS", 50))
```

## Deployment Guide

### Local Deployment

The application runs locally as a Python application with minimal requirements.

### Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t Text-To-Search:latest .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8080:8080 Text-To-Search:latest
   ```

### Cloud Deployment (AWS Example)

1. Push the Docker image to Amazon ECR
2. Deploy using AWS Elastic Beanstalk or ECS
3. Configure the necessary security groups and IAM roles

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Index not loading | Check file permissions and path correctness |
| Slow search performance | Reduce corpus size or optimize index parameters |
| Memory errors | Increase available memory or implement incremental indexing |
| Text encoding issues | Ensure all documents use UTF-8 encoding |

## Security Best Practices

- **Input Validation**: All user queries are sanitized to prevent injection attacks
- **Authentication**: Implement authentication for multi-user deployments
- **Rate Limiting**: Consider adding rate limiting for public-facing deployments
- **Data Privacy**: Document collections should be properly secured
- **Regular Updates**: Keep dependencies updated to patch security vulnerabilities

## Contributing Guidelines

I welcome contributions to Text-To-Search! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add some amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a pull request**

### Contribution Areas

- Bug fixes and improvements
- Additional retrieval models
- Performance optimizations
- Documentation enhancements
- User interface improvements

## Documentation

- API Reference
- Indexing Guide
- Query Syntax
- Performance Tuning

## Roadmap

- **Short-term Goals**:
  - Implement phrase completion and suggestions
  - Add relevance feedback mechanisms
  - Improve documentation coverage

- **Medium-term Goals**:
  - Implement probabilistic retrieval models (BM25)
  - Add support for non-text document types
  - Create a web service API

- **Long-term Vision**:
  - Semantic search capabilities
  - Integration with ML-based ranking models
  - Distributed indexing and search

## Acknowledgments & Credits

- NLTK team for their excellent natural language processing tools

## Contact Information

For questions, suggestions, or collaboration opportunities:

- **GitHub**: [muhammadhamzagova666](https://github.com/muhammadhamzagova666)
- **Project Issues**: Please use the [GitHub issue tracker](https://github.com/muhammadhamzagova666/information-retrieval-system/issues)
