# Boolean Information Retrieval System

Welcome to the Boolean Information Retrieval (IR) system! This tool allows you to search through a collection of research papers using Boolean queries, which can include complex expressions with operators like AND, OR, NOT, and proximity searches.

## Prerequisites

Before running the IR system, ensure you have the following:

- Python 3.12.0 installed on your system.
- NLTK library for Python, which can be installed using the command `pip install nltk`.

## Installation

To run the IR system, you need to install some Python libraries. Open your command prompt or terminal and execute the following commands:

```bash
pip install nltk
pip install json
pip install re
pip install os
pip install collections
pip install tkinter
```

## Setup

1. **Download the Code**: Save all the `.py` files for the IR system in a single folder on your computer.
2. **Prepare the Data**: Place your research paper text files in a folder named `ResearchPapers`. Each paper should be a `.txt` file, named sequentially (e.g., `1.txt`, `2.txt`, `3.txt`, etc.).

## Running the IR System

1. Open your command prompt or terminal.
2. Navigate to the folder containing the `.py` files.
3. Run the command `python main.py` to start the system.

A graphical user interface (GUI) should appear, indicating that the system is ready for use.

## Using the IR System

- **Enter a Query**: Click into the text box and type your search query using the following operators:
  - `AND`: To find documents containing all terms (e.g., `python AND programming`).
  - `OR`: To find documents containing any of the terms (e.g., `python OR programming`).
  - `NOT`: To exclude documents containing the term (e.g., `NOT python`).
  - Proximity Search: To find documents where two terms are within a certain number of words from each other (e.g., `python programming /5`).

- **Run the Query**: Click 'Search'.
- **View Results**: The matching document IDs will be displayed below the search bar.

## Troubleshooting

If you encounter issues:

- Ensure all text files are correctly named and placed in the `ResearchPapers` folder.
- Verify that Python and NLTK are properly installed.
- For more assistance, refer to the documentation provided with the code or contact the developer.
