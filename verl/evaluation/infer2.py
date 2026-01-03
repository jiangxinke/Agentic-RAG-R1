import argparse
import re
from typing import Optional, Tuple, Dict, Any

import sglang as sgl
import os
from dotenv import load_dotenv      

load_dotenv('evaluation/.env')   
from elasticsearch import Elasticsearch
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

def semantic_search(query, index_name, num_results=3):
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
    # print(f"Search query = {query}\n============\n")
    for doc in results:
        res = res + f"{id}. " + doc["text"][:512] + "..."
        id += 1
    return res

_TAG_THINK = "<think>"
_TAG_REFLECT = "<reflect>"
_TAG_TOOLCALL = "<tool_call>"
_TAG_ANSWER = "<answer>"

_END_REFLECT = "</reflect>"
_END_TOOLCALL = "</tool_call>"
_END_ANSWER = "</answer>"


def _extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return m.group(1) if m else ""


def _apply_reflect_deletion(proc_text: str) -> str:
    end_pos = proc_text.rfind(_END_REFLECT)
    if end_pos < 0:
        return proc_text
    
    end_pos2 = end_pos + len(_END_REFLECT)
    truncated = proc_text[:end_pos2]

    reflect_pos = truncated.rfind(_TAG_REFLECT, 0, end_pos2)
    if reflect_pos < 0:
        return truncated

    candidates = [
        truncated.rfind(_TAG_THINK, 0, reflect_pos),
        truncated.rfind(_TAG_REFLECT, 0, reflect_pos),
        truncated.rfind(_TAG_TOOLCALL, 0, reflect_pos),
    ]
    start_pos = max(candidates)
    if start_pos < 0:
        return truncated

    cleaned = truncated[:start_pos] + truncated[reflect_pos:]
    return cleaned

def _extract_last_tool_call_payload(text: str) -> Optional[str]:
    start = text.rfind(_TAG_TOOLCALL)
    if start < 0:
        return None
    end = text.find(_END_TOOLCALL, start)
    if end < 0:
        return None
    return text[start + len(_TAG_TOOLCALL) : end]


def search(proc_concat: str, raw_concat: str) -> Tuple[str, str]:
    payload = _extract_last_tool_call_payload(proc_concat)
    if payload is None:
        return proc_concat, raw_concat

    result = Search(payload)
    tool_resp_block = f"<tool_response>\n{result}\n</tool_response>"
    return proc_concat + tool_resp_block, raw_concat + tool_resp_block


def infer(
    llm: "sgl.Engine",
    prompt: str,
    num_iter: int
) -> Dict[str, Any]:

    proc_concat = prompt
    raw_concat = prompt
    iters = 0
    pre = len(prompt)
    if_tool_call = False

    END_TAGS = [_END_REFLECT, _END_ANSWER, _END_TOOLCALL]
    

    for i in range(num_iter):
        sampling_params = {
            "max_new_tokens": 256,
            "ignore_eos": True,
            # "stop": ["</answer>", "</tool_call>", "</reflect>"],
            # "skip_special_tokens": True,
        }
        # print(f"iter - {i} prompt = {proc_concat} \n ================\n")
        delta = llm.generate([proc_concat+"\n"], sampling_params)[0]["text"]
        
        if delta.strip() == "":
            break
        
        # print (f"before delta = {delta} \n==========\n")

        first_end_pos = None
        first_end_tag = None

        for tag in END_TAGS:
            pos = delta.find(tag)
            if pos != -1:
                if first_end_pos is None or pos < first_end_pos:
                    first_end_pos = pos
                    first_end_tag = tag

        if first_end_pos is not None:
            delta = delta[: first_end_pos + len(first_end_tag)]
            
        # print (f"after delta = {delta} \n==========\n")

        proc_concat += delta
        raw_concat += delta

        if first_end_tag == _END_ANSWER:
            break

        if first_end_tag == _END_TOOLCALL:
            proc_concat, raw_concat = search(proc_concat, raw_concat)
            if_tool_call = True
            continue

        if first_end_tag == _END_REFLECT:
            proc_concat = _apply_reflect_deletion(proc_concat)
            continue
        
        iters = i + 1

    answer = _extract_answer(proc_concat[pre:])

    return {
        "text": proc_concat[pre:],
        "raw_text": raw_concat[pre:],
        "answer": answer,
        "iters": iters,
        "if_tool_call" : if_tool_call
    }

def build_engine(base_model: str, lora_path: Optional[str] = None, device: str = "cuda") -> "sgl.Engine":
    kwargs = {"model_path": base_model, "device": device}
    if lora_path:
        kwargs.update(
            {
                "enable_lora": True,
                "lora_paths": [lora_path],
            }
        )
    return sgl.Engine(**kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="/data2/gjr/models/Qwen2.5-3B-Instruct-spr1-mean-v2", help="Base model path or HF repo id")
    parser.add_argument("--lora", type=str, default=None, help="LoRA adapter path (optional)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu, etc.")
    parser.add_argument("--num-iter", type=int, default=8, help="Max generation iterations")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt string; if omitted, read stdin")

    args = parser.parse_args()

    llm = build_engine(args.base_model, args.lora, device=args.device)
    
    while True:
        prompt = input("query>")
    
        result = infer(
            llm=llm,
            prompt=prompt,
            num_iter=args.num_iter
        )

        print("==== infer result ====")
        print(f"iters: {result['iters']}")
        print("---- answer (inside <answer>...</answer>) ----")
        print(result["answer"])
        print("---- processed text ----")
        print(result["text"])
        print("---- raw text ----")
        print(result["raw_text"])


if __name__ == "__main__":
    main()
