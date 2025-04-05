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


def semantic_search(query, index_name, num_results=10):
    """
    Perform a semantic search on Elasticsearch.

    :param query: user query string.
    :param index_name: Elasticsearch index name.
    :param num_results: number of top results to return.
    :return: list of top search results.
    """
    es = create_es_client()
    search_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title", "text"]
            }
        },
        "size": num_results
    }
    response = es.search(index=index_name, body=search_body)
    
    hits = response['hits']['hits']
    relevant_docs = [hit['_source'] for hit in hits]
    return relevant_docs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Elasticsearch data ingestion pipeline')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'zh'])
    parser.add_argument('--query', type=str, default='Paris 2024 Olympic Games')
    args = parser.parse_args()

    # Test the index with a sample query
    query = args.query
    index_name = f'wiki_{args.language}'
    results = semantic_search(query, index_name)
    for doc in results:
        print(doc)
    