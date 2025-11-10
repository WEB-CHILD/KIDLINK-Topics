#!/usr/bin/env python3
"""
Script to extract all document IDs for a specific topic from the topic model results.

This script loads topic modeling results from a JSON file and displays all document IDs
associated with a specific topic identified by its topic number. The output includes:
- Topic ID and document count
- List of representative document IDs (up to 80 per topic)
- Sample document from the topic (if available)

The document IDs correspond to the 'id' column in the original CSV file and can be used
to look up specific documents in SolrWayback or the source CSV.

The JSON file is expected to contain an array of topic objects, where each object has:
- topic_id: Integer identifier for the topic
- num_docs: Number of documents assigned to this topic
- document_ids: List of document IDs from the CSV
- keywords: List of keywords/terms associated with the topic
- sample: Optional sample document text from the topic

Usage:
    python get_ids_from_topic.py <topic_number> [json_path]
    
Examples:
    python get_ids_from_topic.py 0
    python get_ids_from_topic.py 5 data/topic_model_results.json

Default JSON file: data/topic_model_results.json
"""

import json
import sys
from pathlib import Path


def load_topic_data(json_path: str) -> list:
    """Load topic data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_topic_ids(topic_number: int, json_path: str = None):
    """
    Print all document IDs for a specific topic.
    
    Args:
        topic_number: The topic number to display document IDs for
        json_path: Path to the JSON file (defaults to data/topic_model_results.json)
    """
    if json_path is None:
        # Default to the main topic model results
        project_root = Path(__file__).parent.parent.parent
        json_path = project_root / "data" / "topic_model_results.json"
    
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
    print(f"Top keywords: {', '.join(topic_found.get('keywords', [])[:10])}")
    print(f"\n{'Document IDs:':-<80}")
    
    # Print all document IDs
    document_ids = topic_found.get("document_ids", [])
    if document_ids:
        print(f"Showing {len(document_ids)} document ID(s) (max 80 per topic):")
        print()
        # Print IDs in columns for better readability
        ids_per_row = 5
        for i in range(0, len(document_ids), ids_per_row):
            row_ids = document_ids[i:i+ids_per_row]
            print("  " + "  ".join(f"{doc_id}" for doc_id in row_ids))
    else:
        print("No document IDs found for this topic.")
    
    # Print sample document if available
    sample = topic_found.get("sample", "")
    if sample:
        print(f"\n{'Sample Document:':-<80}")
        # Truncate sample to 300 characters for readability
        sample_preview = sample[:300] + "..." if len(sample) > 300 else sample
        print(sample_preview)
    
    print(f"{'='*80}\n")
    
    # Return the IDs for programmatic use
    return document_ids


def main():
    """Main function to run from command line."""
    if len(sys.argv) < 2:
        print("Usage: python get_ids_from_topic.py <topic_number> [json_path]")
        print("\nExample: python get_ids_from_topic.py 5")
        print("         python get_ids_from_topic.py 5 data/topic_model_results.json")
        sys.exit(1)
    
    try:
        topic_number = int(sys.argv[1])
    except ValueError:
        print(f"Error: Topic number must be an integer, got '{sys.argv[1]}'")
        sys.exit(1)
    
    json_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print_topic_ids(topic_number, json_path)


if __name__ == "__main__":
    main()
