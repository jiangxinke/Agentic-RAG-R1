import os
import glob
import pandas as pd
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from typing import Generator, Dict

def create_es_client() -> Elasticsearch:
    """Initialize Elasticsearch client with environment configuration."""
    if not os.environ.get('ELASTIC_PASSWORD'):
        raise ValueError('ELASTIC_PASSWORD environment variable not set')
    
    return Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD")),
        verify_certs=False,
        ssl_show_warn=False,
    )

def create_es_index(client: Elasticsearch, index_name: str) -> None:
    """Create new Elasticsearch index with cleanup if exists."""
    if client.indices.exists(index=index_name):
        print(f"Index {index_name} already exists. Deleting...")
        client.indices.delete(index=index_name, ignore_unavailable=True)
    client.indices.create(index=index_name)

def process_parquet_file(corpus_file: str, index_name: str) -> Generator[Dict, None, None]:
    """Yield documents from parquet file with progress tracking."""
    df = pd.read_parquet(corpus_file)
    with tqdm(total=len(df), desc=f"Processing {os.path.basename(corpus_file)}") as pbar:
        for _, row in df.iterrows():
            yield {"_index": index_name, **row.to_dict()}
            pbar.update(1)

def index_data(client: Elasticsearch, corpus_files: list, index_name: str) -> None:
    """Bulk index documents with overall progress tracking."""
    total_files = len(corpus_files)
    with tqdm(total=total_files, desc="Total progress") as main_pbar:
        for file_path in corpus_files:
            actions = process_parquet_file(file_path, index_name)
            helpers.bulk(client=client, actions=actions)
            main_pbar.update(1)

def build_elasticsearch(language: str) -> None:
    """Main pipeline for building Elasticsearch index."""
    # Configuration
    index_name = f'wiki_{language}'
    corpus_pattern = f'data/wikipedia/20231101.{language}/*.parquet'
    corpus_files = glob.glob(corpus_pattern)
    
    if not corpus_files:
        raise FileNotFoundError(f"No files found matching pattern: {corpus_pattern}")

    # Initialize Elasticsearch
    client = create_es_client()
    print(f"Connected to cluster: {client.info()['cluster_name']}")
    
    # Index setup
    create_es_index(client, index_name)
    
    # Data ingestion
    print(f"Indexing {len(corpus_files)} files to [{index_name}]")
    index_data(client, corpus_files, index_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Elasticsearch data ingestion pipeline')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'zh'])
    args = parser.parse_args()
    build_elasticsearch(args.language)