import os
import re
import json
from nltk.stem import PorterStemmer
import tkinter as tk
from tkinter import ttk, messagebox
import time
from collections import deque

# Define paths for resources and index files
STOPWORDS_PATH = 'Stopword-List.txt'  # Path to the file containing stopwords.
INVERTED_INDEX_FILE = 'inverted_index.json'  # File to store the inverted index.
POSITIONAL_INDEX_FILE = 'positional_index.json'  # File to store the positional index.

# Helper function to generate file paths for documents
def generate_file_paths(directory, file_count):
    # Creates a list of file paths in the given directory up to the specified file count.
    # Checks if each file exists before adding it to the list.
    return [f'{directory}/{i}.txt' for i in range(1, file_count + 1) if os.path.exists(f'{directory}/{i}.txt')]

# Text pre-processing class
class TextPreprocessor:
    # Initializes the text preprocessor with a path to the stopwords file.
    def __init__(self, stopwords_path):
        self.stopwords = self.load_stopwords(stopwords_path)  # Load the stopwords into memory.
        self.stemmer = PorterStemmer()  # Initialize the Porter Stemmer for stemming words.
        self.token_pattern = re.compile(r"\w+(?:['-,]\w+)?|[^_\w\s]")  # Define a pattern to match words.

    # Loads stopwords from a file and returns them as a list.
    def load_stopwords(self, file_path):
        with open(file_path, 'r', encoding='latin-1') as file:
            return [word.strip() for word in file.readlines()]

    # Tokenizes, stems, and removes stopwords from the text.
    def tokenize_and_stem(self, text):
        text = text.lower()  # Convert text to lowercase.
        tokens = self.token_pattern.findall(text)  # Find all word-like tokens in the text.
        # Stem each token and remove it if it's a stopword or not alphabetic.
        return [self.stemmer.stem(token) for token in tokens if token.isalpha() and token not in self.stopwords]

# Inverted index class
class InvertedIndex:
    # Initializes an empty inverted index.
    def __init__(self):
        self.index = {}

    # Adds tokens to the inverted index under the given document ID.
    def add_tokens(self, document_id, tokens):
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()  # If token not in index, initialize a set for it.
            self.index[token].add(document_id)  # Add the document ID to the set of tokens.

    # Saves the inverted index to a file in JSON format.
    def save(self, file_path):
        # Sort the index and convert sets to lists for JSON serialization.
        sorted_index = {term: sorted(list(postings)) for term, postings in sorted(self.index.items())}
        with open(file_path, 'w') as file:
            json.dump(sorted_index, file, indent=4)  # Write the sorted index to the file.

    # Loads the inverted index from a JSON file.
    def load(self, file_path):
        try:
            with open(file_path, 'r') as file:
                self.index = json.load(file)  # Load the index from the file.
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file: {e}")  # Handle JSON errors.
            self.index = {}  # Reset the index if there's an error.

# Positional index class
class PositionalIndex:
    # Initializes an empty positional index.
    def __init__(self):
        self.index = {}

    # Adds tokens with their positions to the positional index for a given document.
    def add_tokens(self, document_id, tokens):
        for position, token in enumerate(tokens):
            if token not in self.index:
                self.index[token] = {}  # Initialize a dictionary for a new token.
            if document_id not in self.index[token]:
                self.index[token][document_id] = []  # Initialize a list for a new document ID.
            self.index[token][document_id].append(position)  # Append the token position.

    # Saves the positional index to a file in JSON format.
    def save(self, file_path):
        # Sort the index and ensure the positions are sorted for each document.
        sorted_index = {term: {doc_id: positions for doc_id, positions in sorted(postings.items())} for term, postings in sorted(self.index.items())}
        with open(file_path, 'w') as file:
            json.dump(sorted_index, file, indent=4)  # Write the sorted index to the file.

    # Loads the positional index from a JSON file.
    def load(self, file_path):
        try:
            with open(file_path, 'r') as file:
                self.index = json.load(file)  # Load the index from the file.
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file: {e}")  # Handle JSON errors.
            self.index = {}  # Reset the index if there's an error.

# Document processing class
class DocumentProcessor:
    # Initializes the document processor with a path to the stopwords file.
    def __init__(self, stopwords_path):
        self.preprocessor = TextPreprocessor(stopwords_path)  # Create a TextPreprocessor instance.

    # Processes the documents and returns a list of tuples with document IDs and tokens.
    def process_documents(self, file_paths):
        processed_documents = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()  # Read the entire document text.
                tokens = self.preprocessor.tokenize_and_stem(text)  # Tokenize and stem the text.
                document_id = int(os.path.basename(file_path).split('.')[0])  # Extract the document ID from the file name.
                processed_documents.append((document_id, tokens))  # Append the processed document to the list.
        return processed_documents  # Return the list of processed documents.

# Query processing class
class QueryProcessor:
    # Initialization of the QueryProcessor class with necessary indexes and documents.
    def __init__(self, inverted_index, positional_index, all_documents):
        self.inverted_index = inverted_index  # Dictionary for term lookups.
        self.positional_index = positional_index  # Dictionary for term positions within documents.
        self.all_documents = set(all_documents)  # Set of all document IDs to help with 'NOT' queries.
        self.stemmer = PorterStemmer()  # Stemmer to reduce words to their base or root form.

    # Main method to process any given query.
    def process_query(self, query):
        # Check if the query is a proximity query (contains '/').
        if '/' in query:
            # If it is, process it as a proximity query.
            return self.process_proximity_query(query)
        else:
            # Otherwise, process it as a boolean query.
            return self.process_boolean_query(query)

    # Method to process boolean queries.
    def process_boolean_query(self, query):
        # Convert the query to Reverse Polish Notation (RPN) for easier evaluation.
        rpn_query = self.convert_to_rpn(query)
        # Evaluate the RPN query and return the result set.
        return self.evaluate_rpn_query(rpn_query)

    # Method to convert a boolean query into Reverse Polish Notation (RPN).
    def convert_to_rpn(self, query):
        # Initialize an empty deque for the output queue and operator stack.
        output_queue = deque()
        operator_stack = deque()
        # Split the query into individual tokens.
        tokens = self.tokenize_query(query)

        # Define the precedence of each operator.
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1, '(': 0, ')': 0}

        # Iterate over each token in the query.
        for token in tokens:
            if token == '(':
                # Push '(' onto the operator stack.
                operator_stack.append(token)
            elif token == ')':
                # Pop operators from the stack until '(' is encountered.
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                operator_stack.pop()  # Remove the '(' from the stack.
            elif token.upper() in precedence:
                # Pop operators from the stack based on precedence and push onto the output queue.
                while (operator_stack and precedence[operator_stack[-1].upper()] >= precedence[token.upper()]):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            else:
                # Push operands (terms) onto the output queue.
                output_queue.append(token)

        # Pop any remaining operators from the stack onto the output queue.
        while operator_stack:
            output_queue.append(operator_stack.pop())

        # Return the output queue as a list.
        return list(output_queue)

    # Method to evaluate an RPN (postfix) boolean query.
    def evaluate_rpn_query(self, rpn_query):
        # Initialize an empty deque for the stack.
        stack = deque()

        # Iterate over each token in the RPN query.
        for token in rpn_query:
            if token.upper() not in ['AND', 'OR', 'NOT']:
                # If the token is an operand (term), stem it and push its posting list onto the stack.
                stemmed_token = self.stemmer.stem(token.lower())
                stack.append(set(self.inverted_index.get(stemmed_token, [])))
            else:
                # If the token is an operator, pop the necessary number of operands from the stack and apply the operator.
                if token.upper() == 'NOT':
                    term_set = stack.pop()
                    stack.append(self.all_documents - term_set)
                elif token.upper() == 'AND':
                    right_set = stack.pop()
                    left_set = stack.pop()
                    stack.append(left_set & right_set)
                elif token.upper() == 'OR':
                    right_set = stack.pop()
                    left_set = stack.pop()
                    stack.append(left_set | right_set)

        # The final result set is the last item on the stack.
        return sorted(list(stack.pop()), key=int)

    # Method to tokenize the query, handling cases where operators are adjacent to words without spaces.
    def tokenize_query(self, query):
        # Enhanced regular expression to handle operators without spaces
        tokens = re.findall(r'AND|OR|NOT|\(|\)|\w+', query)
        # Stem the tokens that are not operators or parentheses
        return [self.stemmer.stem(token.lower()) if token not in ['AND', 'OR', 'NOT', '(', ')'] else token for token in tokens]

    # Method to process a proximity query and return the set of document IDs where the terms are within the specified proximity.
    def process_proximity_query(self, query):
        # Split the query into parts to identify the terms and the proximity distance.
        parts = query.split()
        # Check for the correct format of a proximity query.
        if len(parts) == 3 and parts[2].startswith('/'):
            # Stem the first and second terms.
            term1 = self.stemmer.stem(parts[0].lower())
            term2 = self.stemmer.stem(parts[1].lower())
            # Extract the proximity distance from the query.
            proximity = int(parts[2][1:])

            # Retrieve the posting lists for both terms from the positional index.
            postings_list1 = self.positional_index.get(term1, {})
            postings_list2 = self.positional_index.get(term2, {})

            # Initialize an empty set for the result.
            result_set = set()
            # Iterate through the documents and positions to find matches within the proximity.
            for doc_id in postings_list1:
                if doc_id in postings_list2:
                    positions1 = postings_list1[doc_id]
                    positions2 = postings_list2[doc_id]
                    # Check each position pair to see if they meet the proximity requirement.
                    for pos1 in positions1:
                        for pos2 in positions2:
                            # If the positions are within the specified proximity, add the document ID to the result set.
                            if abs(pos1 - pos2) - 1 <= proximity:
                                result_set.add(doc_id)
                                break

            # Return the sorted list of document IDs that meet the proximity requirement.
            return sorted(list(result_set), key=int)

# GUI class using Tkinter
class SearchEngineGUI:
    # Initialize the GUI with the main window (root), query processor, and initialization time.
    def __init__(self, root, query_processor, init_time):
        self.root = root  # The main window of the application.
        self.query_processor = query_processor  # Reference to the query processor object.
        self.init_time = init_time  # Time taken to initialize the search engine.
        self.create_widgets()  # Call to create the GUI widgets.

    # Method to create and layout the widgets in the GUI.
    def create_widgets(self):
        self.root.title('Boolean Information Retrieval Model')  # Set the window title.
        style = ttk.Style()  # Create a style object to customize the look of widgets.
        # Configure styles for different widget types.
        style.configure('TLabel', font=('Helvetica', 12))
        style.configure('TButton', font=('Helvetica', 12), background='light blue')
        style.configure('TEntry', font=('Helvetica', 12))
        style.configure('TFrame', background='light gray')

        # Create a frame to hold the entry widget and search button.
        entry_frame = ttk.Frame(self.root, padding="10", style='TFrame')
        entry_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  # Pack the frame at the top of the window.

        # Entry widget for user to type in their search query.
        self.entry = ttk.Entry(entry_frame, width=50)
        self.entry.insert(0, "Enter the query")  # Placeholder text for the entry widget.
        # Bind an event to clear the placeholder text when the entry widget gains focus.
        self.entry.bind("<FocusIn>", lambda args: self.entry.delete('0', 'end') if self.entry.get() == "Enter the query" else None)
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)  # Pack the entry widget to the left.

        # Button that triggers the search operation.
        self.search_button = ttk.Button(entry_frame, text='Search', command=self.perform_search)
        self.search_button.pack(side=tk.RIGHT, padx=5, pady=5)  # Pack the button to the right.

        # Status bar to display initialization time and other messages.
        self.status_bar = ttk.Label(self.root, text=f'Preprocessing Time: {self.init_time:.4f} seconds | Ready', relief=tk.SUNKEN, anchor=tk.W, style='TLabel')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)  # Pack the status bar at the bottom.

        # Label to display the results of the search.
        self.result_label = ttk.Label(self.root, text='', wraplength=400, style='TLabel')
        self.result_label.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)  # Pack the result label below the status bar.

    # Method called when the search button is pressed.
    def perform_search(self):
        query = self.entry.get()  # Get the query from the entry widget.
        # Check if the query is not empty and not the placeholder text.
        if query and query != "Enter the query":
            self.status_bar.config(text='Searching...')  # Update the status bar message.
            self.root.update_idletasks()  # Process all pending events.
            start_time = time.time()  # Record the start time of the search.
            results = self.query_processor.process_query(query)  # Process the query.
            end_time = time.time()  # Record the end time of the search.
            elapsed_time = end_time - start_time  # Calculate the elapsed time.
            # Update the result label with the search results.
            self.result_label.config(text=f'Result Set: {results}')
            # Update the status bar with the preprocessing and query processing times.
            self.status_bar.config(text=f'Preprocessing Time: {self.init_time:.4f} seconds | Query processing time: {elapsed_time:.4f} seconds')
        else:
            messagebox.showinfo('Info', 'Please enter a query.')  # Show an info dialog if the query is empty.

# Main execution function
def main():
    start_time = time.time()  # Record the start time for initialization.
    # Generate file paths for the research papers, assuming there are 30 documents.
    file_paths = generate_file_paths('ResearchPapers', 30)
    # Extract document IDs from the file paths.
    all_documents = [int(os.path.basename(path).split('.')[0]) for path in file_paths]

    # Initialize the document processor with the path to the stopwords file.
    document_processor = DocumentProcessor(STOPWORDS_PATH)
    # Create instances for inverted and positional indexes.
    inverted_index = InvertedIndex()
    positional_index = PositionalIndex()

    # Check if the index files exist. If not, create and save the indexes.
    if not os.path.exists(INVERTED_INDEX_FILE) or not os.path.exists(POSITIONAL_INDEX_FILE):
        # Process the documents to tokenize and stem the text.
        processed_documents = document_processor.process_documents(file_paths)
        for document_id, tokens in processed_documents:
            # Add tokens to both inverted and positional indexes.
            inverted_index.add_tokens(document_id, tokens)
            positional_index.add_tokens(document_id, tokens)
        # Save the indexes to files.
        inverted_index.save(INVERTED_INDEX_FILE)
        positional_index.save(POSITIONAL_INDEX_FILE)
    else:
        # If index files exist, load the indexes from the files.
        inverted_index.load(INVERTED_INDEX_FILE)
        positional_index.load(POSITIONAL_INDEX_FILE)

    # Initialize the query processor with the loaded or created indexes.
    query_processor = QueryProcessor(inverted_index.index, positional_index.index, all_documents)
    end_time = time.time()  # Record the end time for initialization.
    init_time = end_time - start_time  # Calculate the initialization time.

    # Create the main window for the GUI application.
    root = tk.Tk()
    # Initialize the GUI with the root window, query processor, and initialization time.
    app = SearchEngineGUI(root, query_processor, init_time)
    # Start the GUI event loop.
    root.mainloop()

# Check if this script is the main program and run the main function.
if __name__ == "__main__":
    main()