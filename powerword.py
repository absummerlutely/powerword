import csv

import networkx as nx
import pandas as pd

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, DetectorFactory
import yake

# data are scraped from https://www.dimensions.ai/products/all-products/dimensions-free-version/
# Load the dataset
data = pd.read_csv("publications.csv", index_col="title")
data.index.name = None
data["title"] = data.index
print("size of the data:/n", data.shape)

# Preprocess the abstracts
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

additional_stopwords = ['however', 'important', 'also', 'understanding', 'one', 'many', 'several', 'within', 'among','study','area','new','tp','tibetan','plateau','sensitive','region','mountain','used']

# Add domain-specific stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(additional_stopwords)

# remove the abstracts that are empty
data = data[data['abstract'].notna()]

# Set the seed to ensure consistent language detection results
DetectorFactory.seed = 0

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

# Filter out non-English abstracts
data = data[data['abstract'].apply(is_english)]

# Set a similarity threshold to determine which keywords are connected to which publications
similarity_threshold = 0.3

def yake_keywords(text, n_values=(1, 2, 3), top=5):
    all_keywords = []

    for n in n_values:
        kw_extractor = yake.KeywordExtractor(lan="en", n=n, top=top, features=None)
        ignored = stop_words
        keywords = kw_extractor.extract_keywords(text)
        unique_keywords = list(set(kw for kw, score in keywords if kw.lower() not in ignored))
        all_keywords.extend(unique_keywords)

    return all_keywords[:top]


# Create an empty graph
G = nx.Graph()

# Add keyword nodes to the graph
for abstract in data['abstract']:
    keywords = yake_keywords(abstract)
    for keyword in keywords:
        G.add_node(keyword)


# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words=stop_words)
term_document_matrix = vectorizer.fit_transform(data['abstract'])

cosine_sim_matrix = cosine_similarity(term_document_matrix)

# Compute the degree centrality for each keyword
degree_centrality = nx.degree_centrality(G)

keyword_impact = {}
for keyword in G.nodes():
    if isinstance(keyword, str):
        impact = 0
        for pub in G.neighbors(keyword):
            if isinstance(pub, int):
                impact += G.edges[keyword, pub]['weight']
        keyword_impact[keyword] = impact

# Extract the top 10 keywords
sorted_keywords = sorted(keyword_impact.items(), key=lambda x: -x[1])
top_10_keywords = [(keyword, len(list(G.neighbors(keyword))), impact) for keyword, impact in sorted_keywords[:10]]

# Print the top 10 keywords
print("Top 10 keywords:")
for keyword, num_publications, impact in top_10_keywords:
    print(f"{keyword}: impact: {impact}")
