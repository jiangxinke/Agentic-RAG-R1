import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv      
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import argparse

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


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


class Config:
    def __init__(self, index_name: str, retrieval_topk: int = 10):
        self.index_name = index_name
        self.retrieval_topk = retrieval_topk


class ElasticRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.es = create_es_client()

    def batch_search(self, query_list: list[str], num: Optional[int] = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        topk = num or self.config.retrieval_topk
        results = []
        scores = []
        for q in query_list:
            body = {
                "query": {"multi_match": {"query": q, "fields": ["title", "text"]}},
                "size": topk,
            }
            resp = self.es.search(index=self.config.index_name, body=body)
            hits = resp.get("hits", {}).get("hits", [])
            docs = [h.get("_source", {}) for h in hits]
            scs = [h.get("_score", 0.0) for h in hits]
            results.append(docs)
            scores.append(scs)
        if return_score:
            return results, scores
        else:
            return results


app = FastAPI()
config: Config = None
retriever: ElasticRetriever = None


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    if not request.topk:
        request.topk = config.retrieval_topk
    results, score_list = retriever.batch_search(
        query_list=request.queries, num=request.topk, return_score=request.return_scores
    )
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = []
            for doc, score in zip(single_result, score_list[i], strict=True):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Elasticsearch-backed retrieval server.")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--index_name", type=str, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    idx = args.index_name or f"wiki_{args.language}"
    config = Config(index_name=idx, retrieval_topk=args.topk)
    retriever = ElasticRetriever(config)
    uvicorn.run(app, host=args.host, port=args.port)
