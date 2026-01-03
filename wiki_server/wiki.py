import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv      
from rich import print

load_dotenv('.env')   

def create_es_client() -> Elasticsearch:
    """Initialize Elasticsearch client with environment configuration."""
    if not os.environ.get('ELASTIC_PASSWORD'):
        raise ValueError('ELASTIC_PASSWORD environment variable not set')

    return Elasticsearch(
        os.getenv("ELASTIC_URL"),
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


def semantic_search(query, index_name, num_results=10):
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

def Search(query: str) -> str:
    results = semantic_search(query, index_name='wiki_en', num_results=1)
    res = ""
    id = 1
    for doc in results:
        res = res + f"{id}. " + str(doc) + "\n"
        id += 1
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Elasticsearch data ingestion pipeline')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'zh'])
    parser.add_argument('--query', type=str, default='when does the new season of the good dr premiere?')
    args = parser.parse_args()

    # Test the index with a sample query
    index_name = f'wiki_{args.language}'
    # test_index(index_name)
    print (args.query)
    results = semantic_search(args.query, index_name, num_results=3)
    print (Search(args.query))