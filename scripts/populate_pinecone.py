#!/usr/bin/env python3
"""
Populate Pinecone Index - One-time setup script

Embeds all situations and principles from JSON files and upserts to Pinecone.
Run this once after setting up your Pinecone index.

Usage:
    python scripts/populate_pinecone.py
"""

import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    print("=" * 60)
    print("Pinecone Index Population Script")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set in environment")
        print("Set it in .env file or export PINECONE_API_KEY=your_key")
        sys.exit(1)

    index_name = os.getenv("PINECONE_INDEX_NAME", "sales-coach-embeddings")
    print(f"Index name: {index_name}")

    # Import after path setup
    from embeddings import embed_document, EMBEDDING_DIM
    from pinecone_client import PineconeClient

    # Load data files
    project_root = Path(__file__).parent.parent
    situations_path = project_root / "situations.json"
    principles_path = project_root / "principles.json"

    print(f"\nLoading situations from: {situations_path}")
    with open(situations_path) as f:
        situations = json.load(f)
    print(f"  Loaded {len(situations)} situations")

    print(f"\nLoading principles from: {principles_path}")
    with open(principles_path) as f:
        principles_list = json.load(f)
    principles = {p["principle_id"]: p for p in principles_list}
    print(f"  Loaded {len(principles)} principles")

    # Initialize Pinecone client
    print("\nInitializing Pinecone client...")
    client = PineconeClient(api_key=api_key, index_name=index_name)

    # Create index if needed
    print("\nCreating index if it doesn't exist...")
    client.create_index_if_not_exists(dimension=EMBEDDING_DIM)

    # Get current stats
    print("\nCurrent index stats:")
    stats = client.get_stats()
    print(f"  Total vectors: {stats.get('total_vector_count', 0)}")
    for ns, ns_stats in stats.get('namespaces', {}).items():
        print(f"  Namespace '{ns}': {ns_stats.get('vector_count', 0)} vectors")

    # Ask for confirmation before clearing
    print("\n" + "=" * 60)
    response = input("Clear existing data and re-populate? (yes/no): ").strip().lower()
    if response != "yes":
        print("Aborted.")
        sys.exit(0)

    # Clear existing data
    print("\nClearing existing data...")
    try:
        client.delete_all(namespace="situations")
        print("  Cleared 'situations' namespace")
    except Exception as e:
        print(f"  Note: {e}")

    try:
        client.delete_all(namespace="principles")
        print("  Cleared 'principles' namespace")
    except Exception as e:
        print(f"  Note: {e}")

    # Populate situations
    print("\nPopulating situations...")
    print("  (This may take a moment as embeddings are generated)")
    sit_count = client.upsert_situations(situations, embed_document)
    print(f"  Upserted {sit_count} situation vectors")

    # Populate principles
    print("\nPopulating principles...")
    prin_count = client.upsert_principles(principles, embed_document)
    print(f"  Upserted {prin_count} principle vectors")

    # Final stats
    print("\nFinal index stats:")
    stats = client.get_stats()
    print(f"  Total vectors: {stats.get('total_vector_count', 0)}")
    for ns, ns_stats in stats.get('namespaces', {}).items():
        print(f"  Namespace '{ns}': {ns_stats.get('vector_count', 0)} vectors")

    print("\n" + "=" * 60)
    print("Population complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
