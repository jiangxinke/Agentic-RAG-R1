import logging
import os
import re
from multiprocessing.connection import Client
from typing import Dict, List

from src.utils.web_search import web_search
from src.utils.wiki_search import create_wiki_searcher

"""
工具函数

- 首先要在 tools 中添加工具的描述信息
- 然后在 tools 中添加工具的具体实现

- https://serper.dev/dashboard
"""

rag_host = "localhost"
rag_port = 63863


class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()

    def _tools(self):
        tools = [
            # {
            #     'name_for_human': '假设性输出模块',
            #     'name_for_model': 'HypothesisOutput',
            #     'description_for_model': '当你需要简单了解更多相关的知识时，使用这个工具可以得到一些解释，但不一定正确，需要继续使用检索医学知识工具来辅助回答。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '用户询问的字符串',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # },
            # {
            #     'name_for_human': '医学知识抽取模块',
            #     'name_for_model': 'MedicalNER',
            #     'description_for_model': '当需要抽取医学实体时，请使用这个工具。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '需要抽取的字符串形式的医学实体',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # },
            # {
            #     'name_for_human': '医学文档知识检索模块',
            #     'name_for_model': 'DOC_RAG',
            #     'description_for_model': '使用这个工具可以得到医学文档知识，请结合检索的到的部分知识来辅助你回答。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '用户询问的字符串形式的问句',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # },
            # {
            #     'name_for_human': '医学知识图谱路径探查模块',
            #     'name_for_model': 'KG_RAG',
            #     'description_for_model': '使用这个工具可以查询两个医学实体之间的关系，请结合检索的到的部分知识来辅助你回答。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '用户询问的字符串形式的问句',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # },
            {
                "name_for_human": "医学维基百科知识检索模块",
                "name_for_model": "Wiki_RAG",
                "description_for_model": "使用这个工具可以查询百科知识，请结合检索的到的部分知识来辅助你回答。",
                "parameters": [
                    {
                        "name": "input",
                        "description": "规范名称的医学实体",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
            },
            # {
            #     "name_for_human": "医学知识检索模块",
            #     "name_for_model": "Web_RAG",
            #     "description_for_model": "这是通过搜索引擎检索医学知识，请结合检索的到的部分知识来辅助你回答。",
            #     "parameters": [
            #         {
            #             "name": "input",
            #             "description": "用户询问的字符串形式的问句",
            #             "required": True,
            #             "schema": {"type": "string"},
            #         }
            #     ],
            # },
            # {
            #     'name_for_human': '检索的知识总结模块',
            #     'name_for_model': 'KnowledgeOrganize',
            #     'description_for_model': '当检索到的医学知识数量很多时，你可以通过将检索到的医学知识输入，然后用这个工具来做摘要总结。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '需要总结的字符串形式的医学知识',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # },
            # {
            #     'name_for_human': '医学知识过滤模块',
            #     'name_for_model': 'Filter',
            #     'description_for_model': '当需要过滤大量检索到的医学知识中的无关内容时，请使用这个工具。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '需要过滤的字符串形式的医学知识',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # }
        ]
        return tools

    def CircumferenceMahar(self, radius: float) -> float:
        print(radius)
        return radius

    def HypothesisOutput(self, input: str) -> str:
        input = str(input)
        HO_result = send_data(input=input, function_call_type="HO")
        HO_result = str(HO_result)
        HO_result += "但是你后续必须需要检索医学知识辅助你回答，而不是一直使用假设性输出"
        return HO_result if HO_result else "无法探索性回答知识，需要再次调用或规范用户询问"

    def MedicalNER(self, input: str) -> str:
        NER_result = str(send_data(input=input, function_call_type="NER"))
        return NER_result if NER_result else "无法抽取知识，请规范用户输入"

    def DOC_RAG(self, input: str) -> str:
        RAG_result = str(send_data(input=input, function_call_type="DOC"))
        return RAG_result if RAG_result else "无法检索到医学知识，请规范用户输入"

    def KG_RAG(self, input: str) -> str:
        RAG_result = str(send_data(input=input, function_call_type="KG"))
        return RAG_result if RAG_result else "无法检索到医学知识，请规范用户输入"

    def Wiki_RAG(self, input: str) -> str:
        logging.info(f"Using Wiki_RAG input: {input}")
        # wiki_searcher = create_wiki_searcher("zh")
        wiki_searcher = create_wiki_searcher("en")
        assert wiki_searcher is not None
        RAG_result = wiki_searcher.search(input)
        RAG_result = RAG_result if RAG_result else "无法检索到医学知识，请规范用户输入"
        logging.info(f"Wiki_RAG result: {str(RAG_result)[:200]}")
        return str(RAG_result)[:200]

    def Web_RAG(self, input: str) -> str:
        RAG_result = web_search(input, count=1)
        return RAG_result if RAG_result else "无法检索到医学知识，请规范用户输入"

    def KnowledgeOrganize(self, input: str) -> str:
        RAG_result = str(send_data(input=input, function_call_type="KO"))
        return RAG_result if RAG_result else "无法总结医学知识，请规范用户输入"

    def Filter(self, input: str) -> str:
        RAG_result = str(send_data(input=input, function_call_type="Filter"))
        return RAG_result if RAG_result else "无法过滤医学知识，请规范用户输入"


# Helper function to send data
def send_data(input, function_call_type):
    """Sends data to the server and receives the response."""
    try:
        client = Client((rag_host, rag_port))
        data_dict = {"clear": 0, "query": input, "function_call_type": function_call_type}
        client.send(data_dict)
        result = client.recv()  # Wait to receive data
        client.close()
        return result
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    result = send_data("蛋糕", "DOC")
    print(result[0][:50])
