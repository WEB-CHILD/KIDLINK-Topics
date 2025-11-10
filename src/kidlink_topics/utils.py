# Utility functions for text processing and visualization
import pandas as pd

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.corpus import stopwords


def load_csv(file_path):
    """Load documents from a CSV file and return a list of document contents."""
    df = pd.read_csv(file_path)
    documents = df['content'].dropna().tolist()
    print(f"✓ Loaded {len(documents)} documents\n")
    return documents

def load_custom_stopwords():
    """Load and return a combined set of stopwords from various languages and custom file."""
    # Ensure NLTK stopwords are available
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    # Build combined stopword set - Highest priority languages
    danish_sw = set(stopwords.words("danish"))
    norwegian_sw = set(stopwords.words("norwegian"))
    spanish_sw = set(stopwords.words("spanish"))
    english_sw = set(stopwords.words("english"))
    german_sw = set(stopwords.words("german"))
    italian_sw = set(stopwords.words("italian"))
    portuguese_sw = set(stopwords.words("portuguese"))
    french_sw = set(stopwords.words("french"))
    swedish_sw = set(stopwords.words("swedish"))
    dutch_sw = set(stopwords.words("dutch"))
    finnish_sw = set(stopwords.words("finnish"))
    
    # Medium priority additional languages
    russian_sw = set(stopwords.words("russian"))
    turkish_sw = set(stopwords.words("turkish"))
    arabic_sw = set(stopwords.words("arabic"))

    # Load custom domain-specific stopwords from file
    custom_stopwords = set()
    custom_stopwords_file = "data/custom_stopwords.txt"
    try:
        with open(custom_stopwords_file, 'r', encoding='utf-8') as f:
            custom_stopwords = set(line.strip().lower() for line in f if line.strip())
        print(f"  Loaded {len(custom_stopwords)} custom stopwords from {custom_stopwords_file}")
    except FileNotFoundError:
        print(f"  No custom stopwords file found (looked for {custom_stopwords_file})")

    # Combine all stopwords into a single set
    combined_stopwords = set(w.lower() for w in (
        ENGLISH_STOP_WORDS | 
        english_sw | danish_sw | norwegian_sw | spanish_sw | 
        german_sw | italian_sw | portuguese_sw |
        french_sw | swedish_sw | dutch_sw | finnish_sw |
        russian_sw | turkish_sw | arabic_sw
    )) | custom_stopwords 

    return combined_stopwords

def remove_stopwords(text, combined_stopwords):
    """Remove stopwords from text, tokenizing on whitespace."""
    if not isinstance(text, str):
        return ""
    tokens = [t.lower() for t in text.split() if t.lower() and t.lower() not in combined_stopwords]
    filtered = " ".join(tokens)
    return filtered if filtered.strip() else text

def save_array_to_json(array, file_path):
    """Save a Python array to a JSON file."""
    import json
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(array, f, ensure_ascii=False, indent=4)
    print(f"✓ Saved data to {file_path}\n")