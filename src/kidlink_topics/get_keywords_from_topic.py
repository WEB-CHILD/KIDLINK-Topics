#!/usr/bin/env python3
"""
Script to print all keywords for a specific topic from the topic model results.

This script loads topic modeling results from a JSON file and displays all keywords
associated with a specific topic identified by its topic number. The output includes:
- Topic ID and document count
- Complete list of keywords ranked by relevance
- Sample document from the topic (if available)

The JSON file is expected to contain an array of topic objects, where each object has:
- topic_id: Integer identifier for the topic
- num_docs: Number of documents assigned to this topic
- keywords: List of keywords/terms associated with the topic
- sample: Optional sample document text from the topic

Usage:
    python get_keywords_from_topic.py <topic_number> [json_path]
    
Examples:
    python get_keywords_from_topic.py 0
    python get_keywords_from_topic.py 5 data/topic_model_results_30_docs.json

Default JSON file: data/topic_model_results_80_docs_50_keywords.json
"""

import json
import sys
from pathlib import Path


def load_topic_data(json_path: str) -> list:
    """Load topic data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_topic_keywords(topic_number: int, json_path: str = None):
    """
    Print all keywords for a specific topic.
    
    Args:
        topic_number: The topic number to display keywords for
        json_path: Path to the JSON file (defaults to data/topic_model_results_80_docs_50_keywords.json)
    """
    if json_path is None:
        # Default to the 80 docs, 50 keywords version
        project_root = Path(__file__).parent.parent.parent
        json_path = project_root / "data" / "topic_model_results_80_docs_50_keywords.json"
    
    # Load the data
    topics = load_topic_data(json_path)
    
    # Find the topic
    topic_found = None
    for topic in topics:
        if topic.get("topic_id") == topic_number:
            topic_found = topic
            break
    
    if topic_found is None:
        print(f"Error: Topic {topic_number} not found in the data.")
        print(f"Available topics: {sorted([t.get('topic_id') for t in topics if 'topic_id' in t])}")
        return
    
    # Print topic information
    print(f"\n{'='*80}")
    print(f"TOPIC {topic_number}")
    print(f"{'='*80}")
    print(f"Number of documents: {topic_found.get('num_docs', 'N/A')}")
    print(f"\n{'Keywords:':-<80}")
    
    # Print all keywords
    keywords = topic_found.get("keywords", [])
    if keywords:
        for i, keyword in enumerate(keywords, 1):
            print(f"{i:3d}. {keyword}")
    else:
        print("No keywords found for this topic.")
    
    # Print sample document if available
    sample = topic_found.get("sample", "")
    if sample:
        print(f"\n{'Sample Document:':-<80}")
        # Truncate sample to 300 characters for readability
        sample_preview = sample[:300] + "..." if len(sample) > 300 else sample
        print(sample_preview)
    
    print(f"{'='*80}\n")


def main():
    """Main function to run from command line."""
    if len(sys.argv) < 2:
        print("Usage: python get_keywords_from_topic.py <topic_number> [json_path]")
        print("\nExample: python get_keywords_from_topic.py 5")
        print("         python get_keywords_from_topic.py 5 data/topic_model_results_30_docs.json")
        sys.exit(1)
    
    try:
        topic_number = int(sys.argv[1])
    except ValueError:
        print(f"Error: Topic number must be an integer, got '{sys.argv[1]}'")
        sys.exit(1)
    
    json_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print_topic_keywords(topic_number, json_path)


if __name__ == "__main__":
    main()
