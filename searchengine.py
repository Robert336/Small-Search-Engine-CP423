import requests
import hashlib
import datetime
import os
import re
import csv
import sys
import joblib
from joblib import dump
from joblib import load
from soundex import Soundex
from collections import defaultdict
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC


"""
*** REFER TO report.pdf FOR USAGE INSTRUCTIONS AND SETUP ***

"""

def sound_ex(word):
    """
    Computes the Soundex code for a given word
    """
    # Check if the word is empty or contains only non-alphabetic characters
    if not word or not word.isalpha():
        return '0000'

    # Convert word to uppercase
    word = word.upper()

    # Replace consonants with digits according to Soundex
    code = word[0]
    for char in word[1:]:
        if char in 'BFPV':
            code += '1'
        elif char in 'CGJKQSXZ':
            code += '2'
        elif char in 'DT':
            code += '3'
        elif char == 'L':
            code += '4'
        elif char in 'MN':
            code += '5'
        elif char == 'R':
            code += '6'

    # Remove duplicates and zeros from code
    code = ''.join(char for i, char in enumerate(code) if (i == 0 or char != code[i-1]) and char != '0')

    # code with zeros or truncate to 4 digits
    code = code + '0000'
    return code[:4]

MAX_DEPTH = 1

def collect_documents(url, topic, depth):
    """
    Collects new documents from the provided url
    """ 
    if depth > MAX_DEPTH:
        return

    try:
        r = requests.get(url, timeout=3) # timeout after 3 seconds
    except requests.exceptions.RequestException as e:
        print(f"Failed to get content from {url}.")
        return

    soup = BeautifulSoup(r.content, 'html.parser')
    content = soup.get_text()
    content = re.sub(r'\s+', ' ', content).strip()

    if len(content) > 0:
        # Save the content in the topic-related subfolder
        dir_path = f'data/{topic}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # Calculate the hash value of the document content
        doc_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        with open(f'{dir_path}/{doc_hash}.txt', 'w', encoding='utf-8') as f:
            f.write(content)

        # Write the <topic, link’s URL, Hash value of URL, date> in crawl.log file
        with open('crawl.log', 'a') as f:
            f.write(f'{topic}, {url}, {depth}, {doc_hash}, {datetime.datetime.now()}\n')

        # For each link of the crawled page, if the link exactly includes initial URL, crawl it (step 1)
        links = soup.find_all('a', href=True)
        for link in links:
            if url in link['href']:
                collect_documents(link['href'], topic, depth+1)
        
    
def index_documents():
    """
    Creates an inverted index using the downloaded pages and saves it as invertedindex.txt
    """
    inverted_index = defaultdict(list)
    mapping = {}
    N = 0

    for topic in os.listdir('data'):
        dir_path = f'data/{topic}/'
        for doc in os.listdir(dir_path):
            N += 1
            with open(f'{dir_path}/{doc}', 'r', encoding='utf-8') as f:
                content = f.read()
            # Calculate the hash value of the document content
            doc_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

            # Add a new record to the mapping file
            mapping[doc_hash] = f'H{N}'

            # Update the inverted index
            words = re.findall(r'\w+', content)
            for word in words:
                # Add the appearance of the word in the document to the inverted index
                if not any(doc_id == doc_hash for doc_id, _ in inverted_index[word.lower()]):
                    inverted_index[word.lower()].append((doc_hash, words.count(word)))

    # Save the inverted index as invertedindex.txt
    with open('invertedindex.txt', 'w', encoding='utf-8') as f:
        for term, postings in inverted_index.items():
            # Calculate the Soundex code of the term
            soundex_code = sound_ex(term)

            line = f'| {term} | {soundex_code} | '
            for doc_id, tf in postings:
                # Check if the doc_id exists in the mapping dictionary
                if doc_id not in mapping:
                    print(f"Error: DocID {doc_id} not found in mapping.")
                    continue
                # Replace the document hash with its corresponding DocID
                line += f'({mapping[doc_id]}, {tf}) '
            line = line[:-1] + ' |\n'
            f.write(line)

    # Save the mapping file as mapping.txt
    with open('mapping.txt', 'w', encoding='utf-8') as f:
        for doc_hash, doc_id in mapping.items():
            f.write(f'{doc_id},{doc_hash}\n')

    print()
    print("----------------")
    print("Done Indexing")
    print("----------------")
    main()

def search_query():
    """
    Searches for a query and returns the top 3 most related documents
    """
    soundex_words = {}
    inverted_index = defaultdict(list)
    with open('invertedindex.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split(' | ')
            values[0] = values[0].strip('| ')
            values[2] = values[2].strip(' |')

            if len(values) < 2:
                continue  # Skip lines that don't have at least two values
            term = values[0].strip('| ')
            postings = values[2].strip()[1:-1].split(') (')

            # Add soundex as key and the term as value
            if values[1] not in soundex_words:
                soundex_words[values[1]] = [values[0]]
            # Soundex already added, add new term with same soundex
            else:
                soundex_words[values[1]].append(values[0])

            for posting in postings:
                doc_id, tf = posting.strip().split(', ')
                try:
                    tf = int(tf)
                except ValueError:
                    continue  # Skip this posting if tf is not a valid integer
                inverted_index[term].append((doc_id, tf))

    print('Inverted index verified successfully!')
    
    # Load the mapping file
    mapping = {}
    with open('mapping.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split(',')
            if len(values) < 2:
                continue
            doc_hash, topic = values
            mapping[doc_hash] = topic

    # Get the query from the user input
    query = input('Enter your query: ')
    #query = query.lower()
    query_words = query.split()

    # Find list of documents that include at least one of the query terms
    
    doc_scores = defaultdict(float)
    # go through each word in the query
    for term in query_words:
        # check if the term is not present in the inverted index
        if term.lower() not in inverted_index:
            # convert term to soundex if the original term was not present
            soundex_code = sound_ex(term.lower())
            # check if term's soundex is in our dictionary of soundex codes
            if soundex_code in soundex_words:
                # get the lsit of terms that the soundex represents
                soundex_terms = soundex_words[soundex_code]
                max_term = ''
                max_docs = 0
                # Searching through documents with each term
                for term in soundex_terms:
                    if term in inverted_index:
                        term_info = inverted_index[term]
                        if len(term_info) > max_docs:
                            max_docs = len(term_info)
                            max_term = term

                if max_term in inverted_index:
                    term_info = inverted_index[max_term]

                    for doc_id, tf in term_info:
                        if doc_id in doc_scores:
                            doc_scores[doc_id] += tf
                        else:
                            doc_scores[doc_id] = tf
        else: # word is present in the inverted index
            doc_scores = defaultdict(float)
            for term in query_words:
                for doc_id, tf in inverted_index[term.lower()]:
                    if doc_id in doc_scores:
                        doc_scores[doc_id] += tf
                    else:
                        doc_scores[doc_id] = tf
    
    # Vectorize the documents and the query
    X = []
    y = []
    for topic in os.listdir('data'):
        dir_path = f'data/{topic}/'
        for doc in os.listdir(dir_path):
            with open(f'{dir_path}/{doc}', 'r', encoding='utf-8') as f:
                content = f.read()
            X.append(content)
            y.append(topic)

    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(X)
    X_query = tfidf.transform([query])

   # Calculate Cosine similarity between query vector and each document's vector
    similarity_scores = {}
    for doc_id, score in doc_scores.items():
        #doc_id_int = mapping.get(doc_id)
        if doc_id is not None:
            similarity_scores[doc_id] = X[int(doc_id[1:])-1].dot(X_query.T).toarray()[0][0]

    # Rank the documents by score and print the top 3
    sorted_docs = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    if len(sorted_docs) == 0:
        print("No related documents found.")
    else:
        print('Top 3 Results:')
        results = []
        for doc_id, score in sorted_docs:  
            with open('crawl.log', 'r') as log_file:
                log_reader = csv.reader(log_file)
                for row in log_reader:
                    try: #Skips error in crawl log
                        new_var = row[3].strip()
                        if new_var == mapping[doc_id]:
                            row = [s.strip() for s in row]
                            results.append(row)
                            break
                    except:
                        continue

            print(results[-1][1])
            print(f'Category: {results[-1][0]}, Similarity: {score:.4f}, Mapping ID: {doc_id}, Hash: {mapping[doc_id]}')
            
                
    print()
    print("----------------")
    print("Done Query Search")
    print("----------------")
    main()

def train_classifier():
    """
    Trains a classifier using the collected information and saves it as classifier.model
    """
    X = []
    y = []
    for topic in os.listdir('data'):
        dir_path = f'data/{topic}/'
        for doc in os.listdir(dir_path):
            with open(f'{dir_path}/{doc}', 'r', encoding='utf-8') as f:
                content = f.read()
            X.append(content)
            y.append(topic)

    # Vectorize the documents
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the classifier
    clf = SVC(probability=True)
    clf.fit(X_train, y_train)

    # Save the classifier as classifier.model and vectorizer
    joblib.dump(clf, 'classifier.model')
    dump(tfidf, 'tfidf_vectorizer.joblib')

    # Print the training results
    y_pred = clf.predict(X_test)
    print()
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Confusion matrix:\n{confusion_matrix(y_test, y_pred)}')
    
    print()
    print("------------------")
    print("Classifer Trained")
    print("------------------")
    print()
    main()

def predict_link():
    """
    Predicts the topic of a given link using the trained classifier
    """
    link = input('Enter the link to predict: ')
    r = requests.get(link)
    soup = BeautifulSoup(r.content, 'html.parser')
    content = soup.get_text()
    content = re.sub(r'\s+', ' ', content).strip()

    # Load the saved TfidfVectorizer
    tfidf = load('tfidf_vectorizer.joblib')

    # Vectorize the link
    X_link = tfidf.transform([content])

    # Load the saved classifier
    clf = joblib.load('classifier.model')

    # Predict the topic and its confidence score
    topic = clf.predict(X_link)[0]
    confidence = clf.predict_proba(X_link)[0][list(clf.classes_).index(topic)]

    print(f'{topic} - {confidence*100:.2f}%')

    print()
    print("----------------")
    print("   Predicted")
    print("----------------")
    main()

def your_story():
    """
    Prints your story.txt file
    """
    with open('story.txt', 'r') as f:
        story = f.read()
    print()
    print(story)
    print()
    print("----------------")
    print("The End")
    print("----------------")
    main()

def main():
    """
    Displays the options for the user to choose and executes the selected task
    """
    print()
    print('Select an option:')
    print()
    print('1- Collect new documents.')
    print('2- Index documents.')
    print('3- Search for a query.')
    print('4- Train ML classifier.')
    print('5- Predict a link.')
    print('6- Your story!')
    print('7- Exit.')
    print()
    option = input("Your Choice: ")
    if option == '1':
        with open('sources.txt', 'r') as f:
            reader = csv.reader(f)
            sources = list(reader)
            for source in sources:
                topic = source[0]
                url = source[1].strip()
                collect_documents(url, topic, 0)

        print()
        print("----------------")
        print("Done Crawling")
        print("----------------")
        main()
                
    elif option == '2':
        index_documents()
    elif option == '3':
        search_query()
    elif option == '4':
        train_classifier()
    elif option == '5':
        predict_link()
    elif option == '6':
        your_story()
    elif option == '7':
        print("----------------")
        print("Goodbye")
        print("----------------")
        print()
        sys.exit()
    else:
        print('Invalid option!')
        main()

if __name__ == '__main__':
    main()