import os

import elasticsearch
import pytest
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()


class WikiSearcher:
    """
    A class to interact with an Elasticsearch index for performing search queries.
    """

    def __init__(self, index_name="wiki_en", url: str = "https://localhost:9200"):
        """
        Initializes the ElasticsearchSearcher with the given configuration.

        Args:
            index_name (str): Name of the Elasticsearch index.
            hostname (str, optional): Elasticsearch hostname. Defaults to 'localhost'.
            port (int, optional): Elasticsearch port. Defaults to 9200.
            size (int, optional): Number of search results to return. Defaults to 10.
        """

        self.index_name = index_name
        self.es = self.create_es_client(url)

        # Verify connection
        if not self.es.ping():
            raise ConnectionError(f"Cannot connect to Elasticsearch at {url}")

    def search(self, query, size=1):
        """
        Searches the Elasticsearch index for documents matching the query.

        Args:
            query (str): The search query string.

        Returns:
            list: A list of dictionaries containing search results.
        """
        # Define the search query using multi_match to search in 'title' and 'text' fields
        search_body = {"query": {"multi_match": {"query": query, "fields": ["title", "text"]}}, "size": size}

        # Execute the search
        response = self.es.search(index=self.index_name, body=search_body)

        hits = response["hits"]["hits"]
        relevant_docs = [hit["_source"] for hit in hits]
        return relevant_docs

    def close(self):
        """
        Closes the Elasticsearch connection.
        """
        self.es.close()

    @staticmethod
    def create_es_client(url: str) -> Elasticsearch:
        """Initialize Elasticsearch client with environment configuration."""
        if not os.environ.get("ELASTIC_SEARCH_PASSWORD"):
            raise ValueError("ELASTIC_SEARCH_PASSWORD environment variable not set")

        return Elasticsearch(
            url,
            basic_auth=("elastic", os.getenv("ELASTIC_SEARCH_PASSWORD")),
            verify_certs=False,
            ssl_show_warn=False,
        )


def test_english_wiki_searcher():
    wiki_searcher = create_wiki_searcher("en")
    assert wiki_searcher is not None
    result = wiki_searcher.search("why is the sky blue?")
    assert isinstance(result, list)


def test_chinese_wiki_searcher():
    wiki_searcher = create_wiki_searcher("zh")
    assert wiki_searcher is not None
    result = wiki_searcher.search("为什么天空是蓝色的？")
    assert isinstance(result, list)


def test_unsupported_language():
    wiki_searcher = create_wiki_searcher("unsupported")
    assert wiki_searcher is not None
    with pytest.raises(elasticsearch.NotFoundError):
        wiki_searcher.search("why is the sky blue?")


def create_wiki_searcher(language: str, url: str = None) -> WikiSearcher:
    index_name = "wiki_" + language.lower()
    url = url or os.environ.get("ELASTIC_SEARCH_URL")
    if url:
        return WikiSearcher(index_name, url)
    else:
        return WikiSearcher(index_name)


def main():
    """
    Main function to handle command-line arguments and perform searches.
    """
    # Parse command-line arguments
    query = "Paris 2024 Olympic Games"

    try:
        # Initialize the searcher
        searcher = WikiSearcher()

        # Perform the search
        results = searcher.search(query)

        # Display the results
        if not results:
            print("No relevant documents found.")
        else:
            for idx, doc in enumerate(results, start=1):
                print(f"content #{idx}:")
                print(f"{doc}\n")

    except ConnectionError as ce:
        print(f"Connection Error: {ce}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Ensure the Elasticsearch connection is closed
        if "searcher" in locals():
            searcher.close()


if __name__ == "__main__":
    main()
