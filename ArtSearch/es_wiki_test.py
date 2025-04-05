import os
from elasticsearch import Elasticsearch


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


def test_index(index_name):
    es = create_es_client()
    try:
        # Count the number of documents in the specified index
        count_response = es.count(index=index_name)
        doc_count = count_response['count']
        print(f"Number of documents in index {index_name}: {doc_count}")

        # Display some example documents
        search_response = es.search(index=index_name, size=2)
        hits = search_response['hits']['hits']
        if hits:
            print("Example documents:")
            for hit in hits:
                print(hit['_source'])
        else:
            print("No example documents found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Elasticsearch data ingestion pipeline')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'zh'])
    args = parser.parse_args()

    # Test the index with a sample query
    index_name = f'wiki_{args.language}'
    test_index(index_name)
    