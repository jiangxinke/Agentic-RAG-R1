import os
import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from typing import List
from openai import OpenAI
import asyncio

# -----------------------------
# 异步 LLM 相关性评分器
# -----------------------------
class AsyncLLMRelevanceScorer:
    """
    异步安全 LLM 评分器，替代 RAGAS ContextRelevance。
    输入 query + List[content]，输出平均相关性 [0.0, 1.0]。
    使用 OpenAI 官方 SDK。
    """

    def __init__(self, model: str, api_key: str, base_url: str = None, max_workers: int = 2):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # -----------------------------
    # 同步单条评分
    # -----------------------------
    def _score_single(self, query: str, content: str) -> float:
        try:
            prompt = f"""
You are a content relevance scoring expert.
Based on the query and content below, assign a relevance score to the content (ranging from 0.0 to 1.0):
Query: {query}
Content: {content}
Return only a single floating-point number.

NOTE: A value closer to 1 indicates higher relevance, while a value closer to 0 indicates lower relevance.
"""
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content
            score = float(text.strip())
            print(f"LLM relevance score for content: {score}")
            if math.isnan(score) or score < 0.0 or score > 1.0:
                return 0.0
            return score
        except Exception as e:
            print(f"Error scoring content: {content}. Exception: {e}")
            return 0.0

    # -----------------------------
    # 异步单条评分
    # -----------------------------
    async def _score_single_async(self, query: str, content: str) -> float:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._score_single, query, content)

    # -----------------------------
    # 批量评分
    # -----------------------------
    async def score_batch(self, query: str, contents: List[str]) -> float:
        if not contents:
            print (f"Empty contents for query: {query}")
            return 0.0
        tasks = [self._score_single_async(query, c) for c in contents]
        scores = await asyncio.gather(*tasks)
        return sum(scores) / len(scores)


# -----------------------------
# 使用示例
# -----------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    query = "Where was the movie filmed that Simon Curtis played Royce Du Lac in?"
    contents = ['{"name": "search", "arguments": {"query_list": ["Simon Curtis Royce Du Lac movie name"]}}']

    scorer = AsyncLLMRelevanceScorer(
        model=os.getenv("EVAL_LLM_MODEL_NAME"),
        api_key=os.getenv("EVAL_LLM_API_KEY"),
        base_url=os.getenv("EVAL_LLM_BASE_URL"),
        max_workers=2,
    )

    async def test():
        score = await scorer.score_batch(query, contents)
        print("LLM relevance score =", score)

    asyncio.run(test())
