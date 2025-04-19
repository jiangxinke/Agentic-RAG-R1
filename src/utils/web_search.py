import json
import logging
import os

import requests


class BochaWebSearcher:
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("BOCHA_API_KEY")
            assert api_key is not None, "BOCHA_API_KEY is not set"

        self.api_key = api_key

    def search(self, query: str, count: int = 10) -> list[dict]:

        url = os.getenv("BOCHA_APU_URL")

        payload = json.dumps(
            {
                "query": query,
                "freshness": "noLimit",
                "summary": True,
                "count": count,
                "page": 1,
            }
        )

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()

        results = self._parse_response(response.json())

        return [
            {"link": result["url"], "title": result.get("name"), "snippet": result.get("summary")}
            for result in results.get("webpage", [])[:count]
        ]

    @staticmethod
    def _parse_response(response: dict):
        result = {}
        if "data" in response:
            data = response["data"]
            if "webPages" in data:
                webPages = data["webPages"]
                if "value" in webPages:
                    result["webpage"] = [
                        {
                            "id": item.get("id", ""),
                            "name": item.get("name", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("snippet", ""),
                            "summary": item.get("summary", ""),
                            "siteName": item.get("siteName", ""),
                            "siteIcon": item.get("siteIcon", ""),
                            "datePublished": item.get("datePublished", "") or item.get("dateLastCrawled", ""),
                        }
                        for item in webPages["value"]
                    ]
        return result


def web_search(query: str, count: int = 10):
    searcher = BochaWebSearcher()
    logging.info(f"Searching ...")
    return searcher.search(query, count)


if __name__ == "__main__":
    search_result = str(web_search("肚子疼怎么办？", count=1))
    print(type(search_result))
    print(search_result)
