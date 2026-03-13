# Utility functions for text processing and visualization
import pandas as pd

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.corpus import stopwords


def load_csv(file_path):
    """Load documents from a CSV file and return a list of document contents and their IDs."""
    df = pd.read_csv(file_path)
    # Remove rows where content is NaN
    df_clean = df.dropna(subset=['content'])
    documents = df_clean['content'].tolist()
    # Get IDs from the 'id' column, or use index if no 'id' column exists
    if 'id' in df_clean.columns:
        doc_ids = df_clean['id'].tolist()
    else:
        doc_ids = df_clean.index.tolist()
    print(f"✓ Loaded {len(documents)} documents\n")
    return documents, doc_ids

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

def save_figure(filepath, fig, **kwargs):
    """Save matplotlib figure with automatic directory creation and overwrite warning.
    
    Args:
        filepath: Path where figure should be saved
        fig: Matplotlib figure object
        **kwargs: Additional arguments to pass to fig.savefig()
    """
    import os
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    if os.path.exists(filepath):
        print(f"⚠ Overwriting existing file: {filepath}")
    fig.savefig(filepath, **kwargs)

def get_device():
    """Detect and return the best available device for PyTorch computation.
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    import torch
    
    # Check for NVIDIA CUDA (Linux/Windows with NVIDIA GPU)
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ Using NVIDIA GPU: {device_name}\n")
        return device
    
    # Check for Apple Silicon GPU (macOS with M1/M2/M3/M4)
    if torch.backends.mps.is_available():
        device = "mps"
        print("✓ Using Apple Silicon GPU (MPS) for acceleration\n")
        return device
    
    # Fall back to CPU
    device = "cpu"
    print("⚠ No GPU available, using CPU\n")
    return device

def get_output_paths(domain_prefix=None):
    """Generate output file paths with optional domain prefix.
    
    Args:
        domain_prefix (str, optional): Domain prefix to add to filenames.
        If None, uses default paths.
    
    Returns:
        tuple: (output_json_path, output_model_path)
    """
    json_path = get_topic_results_path(domain_prefix)
    if domain_prefix:
        model_path = f"models/{domain_prefix}_topic_model_bertopic"
    else:
        model_path = "models/topic_model_bertopic"
    
    return json_path, model_path

def get_topic_results_path(domain_prefix=None):
    """Generate the path to topic model results JSON file.
    
    Args:
        domain_prefix (str, optional): Domain prefix to add to filename.
        If None, uses default path.
    
    Returns:
        str: Path to topic_model_results.json file
    """
    if domain_prefix:
        return f"data/{domain_prefix}_topic_model_results.json"
    else:
        return "data/topic_model_results.json"

