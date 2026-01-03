import asyncio
import copy
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4
from verl.spr1_generation.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

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
    if query == "":
        return ""
    results = semantic_search(query, index_name='wiki_en', num_results=1)
    res = ""
    id = 1
    for doc in results:
        text = doc.get("text", "")
        if text and len(text) > 256:
            text = text[:256] + "...(truncated)"
        res = res + f"{id}. " + text + "\n"
        id += 1
    return res


@dataclass
class _OpSpan:
    name: str
    start: int
    end: Optional[int] = None


# -----------------------------
# token-seq helpers (NO single-token assumption)
# -----------------------------
def _dedup_seqs(seqs: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for s in seqs:
        t = tuple(s)
        if t not in seen:
            out.append(s)
            seen.add(t)
    return out


def _find_subseq_first(hay: list[int], needle: list[int], start: int = 0) -> int:
    m = len(needle)
    if m == 0:
        return start
    n = len(hay)
    if start < 0:
        start = 0
    if start + m > n:
        return -1
    for i in range(start, n - m + 1):
        if hay[i : i + m] == needle:
            return i
    return -1


def _find_any_first(hay: list[int], needles: list[list[int]], start: int = 0) -> Optional[tuple[int, int, int]]:
    best: Optional[tuple[int, int, int]] = None
    for j, nd in enumerate(needles):
        i = _find_subseq_first(hay, nd, start=start)
        if i == -1:
            continue
        cand = (i, i + len(nd), j)
        if best is None or cand[0] < best[0]:
            best = cand
    return best


def _endswith(hay: list[int], suffix: list[int]) -> bool:
    m = len(suffix)
    return m > 0 and len(hay) >= m and hay[-m:] == suffix


def _strip_suffix(hay: list[int], suffix: list[int]) -> list[int]:
    if _endswith(hay, suffix):
        return hay[: -len(suffix)]
    return hay


def _strip_suffix_any(hay: list[int], suffixes: list[list[int]]) -> tuple[list[int], Optional[list[int]]]:
    for sfx in suffixes:
        if _endswith(hay, sfx):
            return hay[: -len(sfx)], sfx
    return hay, None


# -----------------------------
# path tag cleanup (text-level robust)
# -----------------------------
def _clean_path_tags_text(s: str) -> str:
    """
    Remove any [path*], [/path*], <path*>, </path*> tags from model-generated text.
    """
    s = re.sub(r"\[/?path[^\]]*\]", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</?path[^>]*>", "", s, flags=re.IGNORECASE)
    return s


@register("spr1_agent")
class SPR1AgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor=None, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level SPR1AgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor

        # -------- config robustness --------
        rollout_cfg = None
        try:
            rollout_cfg = config.actor_rollout_ref.rollout
        except Exception:
            rollout_cfg = None

        def _get_rollout_attr(name: str, default):
            if rollout_cfg is None:
                return default
            try:
                return getattr(rollout_cfg, name)
            except Exception:
                try:
                    return rollout_cfg.get(name, default)
                except Exception:
                    return default

        cls.num_paths = int(_get_rollout_attr("spr1_parallel_search_num_paths", 3))
        cls.max_path_response_length = int(_get_rollout_attr("spr1_max_one_step_length", 128))
        cls.max_turns = int(_get_rollout_attr("spr1_max_num_turns", 10))

        data_cfg = {}
        try:
            data_cfg = config.data
        except Exception:
            data_cfg = {}
        # try:
        #     cls.apply_chat_template_kwargs = dict(data_cfg.get("apply_chat_template_kwargs", {}))
        # except Exception:
        cls.apply_chat_template_kwargs = {}
            
        cls.prompt_length = data_cfg.get("max_prompt_length", 4096)
        cls.response_length = data_cfg.get("max_response_length", 2048)

        # avoid: apply_chat_template got multiple values for add_generation_prompt
        cls.apply_chat_template_kwargs.pop("add_generation_prompt", None)

        cls.EOS = cls.tokenizer.eos_token_id

        def _enc(s: str) -> list[int]:
            return cls.tokenizer.encode(s, add_special_tokens=False)

        def _variants(tag: str) -> list[list[int]]:
            """
            Robust variants: BOTH leading and trailing whitespace.
            This fixes the 'did not stop at </reflect>' issue when model outputs '</reflect>\\n'.
            """
            prefixes = ["", "\n", " ", "\n\n"]
            suffixes = ["", "\n", " ", "\n\n"]
            seqs = []
            for pre in prefixes:
                for suf in suffixes:
                    ids = _enc(pre + tag + suf)
                    if ids:
                        seqs.append(ids)
            return _dedup_seqs(seqs)

        cls.NEWLINE = _enc("\n")  # may be multi-token
        cls.TAG_TOOL_CALL = _enc("<tool_call>")
        cls.TAG_TOOL_CALL_END = _enc("</tool_call>")
        cls.TAG_TOOL_RESP = _enc("<tool_response>")
        cls.TAG_TOOL_RESP_END = _enc("</tool_response>")
        cls.TAG_THINK = _enc("<think>")
        cls.TAG_THINK_END = _enc("</think>")
        cls.TAG_REFLECT = _enc("<reflect>")
        cls.TAG_REFLECT_END = _enc("</reflect>")
        cls.TAG_ANSWER = _enc("<answer>")
        cls.TAG_ANSWER_END = _enc("</answer>")

        # boundary variants we want to stop segments on
        cls.V_TOOL_CALL = _variants("<tool_call>")
        cls.V_REFLECT_END = _variants("</reflect>")     # now matches </reflect>\n etc.
        cls.V_ANSWER_END = _variants("</answer>")

        # for op scanning (start/end tags)
        cls.V_THINK = _variants("<think>")
        cls.V_THINK_END = _variants("</think>")
        cls.V_REFLECT = _variants("<reflect>")
        cls.V_TOOL_CALL_END = _variants("</tool_call>")
        cls.V_TOOL_RESP = _variants("<tool_response>")
        cls.V_TOOL_RESP_END = _variants("</tool_response>")
        cls.V_ANSWER = _variants("<answer>")

        # path prefix stop: stop as soon as we see '[/path' (any suffix)
        cls.PATH_CLOSE_PREFIX = "[/path"
        cls.TAG_PATH_CLOSE_PREFIX = _enc(cls.PATH_CLOSE_PREFIX)
        cls.V_PATH_CLOSE_PREFIX = _variants(cls.PATH_CLOSE_PREFIX)

        cls._MAX_TAG_LEN = max(
            [len(x) for x in (
                cls.V_THINK + cls.V_THINK_END + cls.V_REFLECT + cls.V_REFLECT_END +
                cls.V_TOOL_CALL + cls.V_TOOL_CALL_END + cls.V_TOOL_RESP + cls.V_TOOL_RESP_END +
                cls.V_ANSWER + cls.V_ANSWER_END + cls.V_PATH_CLOSE_PREFIX
            )] + [1]
        )

    @staticmethod
    def _get_token_ids(x) -> list[int]:
        if isinstance(x, list):
            return x
        if hasattr(x, "token_ids"):
            return list(x.token_ids)
        if hasattr(x, "tokens"):
            return list(x.tokens)
        if hasattr(x, "output_ids"):
            return list(x.output_ids)
        raise TypeError(f"Unknown generate output type: {type(x)}")

    # -----------------------------
    # segment generation: chunk + local boundary cut
    # -----------------------------
    async def _generate_segment_until_boundary(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        boundary_needles: list[list[int]],
        max_tokens_total: int,
        chunk_tokens: int = 128,
    ) -> tuple[list[int], Optional[list[int]]]:
        acc: list[int] = []
        sp = copy.deepcopy(sampling_params)
        sp["n"] = 1

        # Only stop on EOS at backend level; all other boundaries are multi-token and handled locally.
        if self.EOS is not None:
            sp["stop_token_ids"] = [self.EOS]
        else:
            sp.pop("stop_token_ids", None)

        remaining = int(max_tokens_total)
        while remaining > 0:
            sp["max_new_tokens"] = int(min(chunk_tokens, remaining))

            out = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids + acc,
                sampling_params=sp,
            )
            new_ids = self._get_token_ids(out)
            if not new_ids:
                break

            # consume EOS if returned (avoid sticky stop)
            if self.EOS is not None and new_ids and new_ids[-1] == self.EOS:
                new_ids = new_ids[:-1]

            base_len = len(acc)
            acc.extend(new_ids)
            remaining -= len(new_ids)

            # search earliest boundary occurrence, starting slightly before new chunk start
            search_from = max(0, base_len - (self._MAX_TAG_LEN + 8))
            hit = _find_any_first(acc, boundary_needles, start=search_from)
            if hit is not None:
                s, e, _ = hit
                acc = acc[:e]
                matched = acc[s:e]
                return acc, matched

            # short output => backend likely stopped early; do not spin
            if len(new_ids) > 0 and len(new_ids) < sp["max_new_tokens"]:
                break
        
        return acc, None

    # -----------------------------
    # one tool_call path generation (strict path handling)
    # -----------------------------
    async def _gen_one_path(
        self,
        *,
        base_prompt: list[int],
        path_index: int,
        sampling_params: dict[str, Any],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """
        REQUIRED behavior (per your request):
          - We inject '[path{i}]' ourselves, so generation starts right after it (model does not generate it).
          - Stop as soon as '[/path' appears (any index), i.e., stop on close-prefix.
          - Remove any '[path*]' or '[/path*]' tags from generated content (j arbitrary).
        Returns (clean_path_body_ids, open_ids, close_ids, newline_ids)
        """
        path_open = f"[path{path_index}]"
        path_close = f"[/path{path_index}]"
        open_ids = self.tokenizer.encode(path_open, add_special_tokens=False)
        close_ids = self.tokenizer.encode(path_close, add_special_tokens=False)
        newline_ids = self.NEWLINE

        # Prompt for this path: base_prompt + open_ids
        path_prompt = base_prompt + open_ids

        # boundary needles: stop immediately once close-prefix appears, or newline, or </tool_call> (leak guard)
        needles: list[list[int]] = []
        needles += self.V_PATH_CLOSE_PREFIX
        needles += self.V_TOOL_CALL_END
        needles += self.V_ANSWER
        needles += self.V_REFLECT
        needles += self.V_TOOL_RESP

        sp = copy.deepcopy(sampling_params)
        sp["n"] = 1
        sp["temperature"] = float(sp.get("temperature", 1.5))
        sp["max_new_tokens"] = int(max(64, self.max_path_response_length))

        seg_ids, _matched = await self._generate_segment_until_boundary(
            request_id=uuid4().hex,
            prompt_ids=path_prompt,
            sampling_params=sp,
            boundary_needles=needles,
            max_tokens_total=sp["max_new_tokens"],
            chunk_tokens=64,
        )

        # If seg ended on close-prefix/newline/tool_call_end, strip suffix variants (best-effort)
        seg_ids, _ = _strip_suffix_any(seg_ids, self.V_PATH_CLOSE_PREFIX)
        if self.NEWLINE:
            seg_ids = _strip_suffix(seg_ids, self.NEWLINE)
        seg_ids, _ = _strip_suffix_any(seg_ids, self.V_TOOL_CALL_END)

        # Remove any [path*] tags the model may have emitted inside seg
        seg_text = self.tokenizer.decode(seg_ids, skip_special_tokens=False)
        seg_text_clean = _clean_path_tags_text(seg_text).strip()
        seg_ids_clean = self.tokenizer.encode(seg_text_clean, add_special_tokens=False)

        return seg_ids_clean, open_ids, close_ids, newline_ids

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        metrics = {}
        request_id = uuid4().hex

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )

        init_len = len(prompt_ids)

        full_ids: list[int] = list(prompt_ids)
        resp_mask: list[int] = []
        pos_ids: list[int] = list(range(init_len))
        position_required_masks: list[list[int]] = []
        op_history: list[_OpSpan] = []
        open_stack: list[_OpSpan] = []

        def _append_ids(new_ids: list[int], gen_by_model: bool) -> tuple[int, int]:
            # text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
            # text = text.replace("\n", "")
            # print(f"[ADD] |||| {text} |||| gen_by_model={gen_by_model}")
            start = len(full_ids)
            full_ids.extend(new_ids)
            resp_mask.extend(([1] if gen_by_model else [0]) * len(new_ids))
            last = pos_ids[-1] if pos_ids else -1
            pos_ids.extend([last + i + 1 for i in range(len(new_ids))])
            return start, start + len(new_ids)

        # -----------------------------
        # op scanner (sequence-based)
        # -----------------------------
        TAG_EVENTS: list[tuple[bool, str, list[list[int]]]] = [
            (True, "think", self.V_THINK),
            (False, "think", self.V_THINK_END),
            (True, "reflect", self.V_REFLECT),
            (False, "reflect", self.V_REFLECT_END),
            (True, "tool_call", self.V_TOOL_CALL),
            (False, "tool_call", self.V_TOOL_CALL_END),
            (True, "tool_response", self.V_TOOL_RESP),
            (False, "tool_response", self.V_TOOL_RESP_END),
            (True, "answer", self.V_ANSWER),
            (False, "answer", self.V_ANSWER_END),
        ]

        EVENT_SEQS: list[list[int]] = []
        EVENT_META: list[tuple[bool, str]] = []
        for is_start, name, seqs in TAG_EVENTS:
            for s in seqs:
                EVENT_SEQS.append(s)
                EVENT_META.append((is_start, name))

        scan_cursor = init_len

        def _scan_ops_until(end_idx: int):
            nonlocal scan_cursor, position_required_masks
            start = max(init_len, scan_cursor - (self._MAX_TAG_LEN + 8))

            while True:
                hit = _find_any_first(full_ids[:end_idx], EVENT_SEQS, start=start)
                if hit is None:
                    scan_cursor = end_idx
                    return

                s, e, j = hit
                is_start, name = EVENT_META[j]

                if is_start:
                    span = _OpSpan(name=name, start=s, end=None)
                    open_stack.append(span)
                    op_history.append(span)
                else:
                    k = len(open_stack) - 1
                    while k >= 0 and open_stack[k].name != name:
                        k -= 1
                    if k >= 0:
                        span = open_stack[k]
                        span.end = e
                        open_stack.pop(k)

                        if name == "reflect":
                            Lreflect = span.start
                            Rreflect_end = e - 1

                            prev_start = None
                            for prev in reversed(op_history):
                                if prev.start >= Lreflect:
                                    continue
                                if prev.name not in ("think", "reflect", "tool_call"):
                                    continue
                                prev_start = prev.start
                                break

                            if prev_start is not None and prev_start <= Lreflect - 1:
                                position_required_masks.append(
                                    [
                                        prev_start,
                                        Lreflect - 1,
                                        (Rreflect_end + 1),
                                        -1,
                                    ]
                                )

                start = e

        # -----------------------------
        # position_ids alignment helpers (unchanged)
        # -----------------------------
        def _align_paths_position_ids(path_spans: list[tuple[int, int]], base_pos: int) -> int:
            longest = 0
            for (s, e) in path_spans:
                if s >= len(pos_ids) or e - 1 >= len(pos_ids):
                    continue
                shift = pos_ids[s] - base_pos
                if shift != 0:
                    for k in range(s, e):
                        pos_ids[k] -= shift
                path_max = pos_ids[e - 1]
                longest = max(longest, path_max - base_pos)
            return int(longest)

        def _shift_after(idx: int, desired_start_pos: int):
            if idx >= len(pos_ids):
                return
            shift = pos_ids[idx] - desired_start_pos
            if shift != 0:
                for k in range(idx, len(pos_ids)):
                    pos_ids[k] -= shift

        # -----------------------------
        # main loop
        # -----------------------------
        iters = 0
        with simple_timer("generate_sequences", metrics):
            base_sp = copy.deepcopy(sampling_params)
            base_max = int(base_sp.get("max_new_tokens", self.response_length))
            base_sp["max_new_tokens"] = max(base_max, self.response_length)

            max_iters = self.max_turns
            chunk_tokens = int(base_sp.get("chunk_tokens", 128))

            # IMPORTANT: segment boundaries
            # - stop on <tool_call>
            # - stop on </reflect>  (now robust even with trailing newline)
            # - stop on </answer>
            boundary_needles = self.V_TOOL_CALL + self.V_REFLECT_END + self.V_ANSWER_END

            while iters < max_iters:
                iters += 1
                seg_ids, matched_seq = await self._generate_segment_until_boundary(
                    request_id=request_id,
                    prompt_ids=full_ids + self.NEWLINE,
                    sampling_params=base_sp,
                    boundary_needles=boundary_needles,
                    max_tokens_total=int(base_sp["max_new_tokens"]),
                    chunk_tokens=chunk_tokens,
                )
                if not seg_ids:
                    break

                seg_s, seg_e = _append_ids(seg_ids, gen_by_model=True)
                _scan_ops_until(seg_e)

                # helper: locate exact tag inside matched (handles leading/trailing ws variants)
                def _locate_exact(exact_tag: list[int], matched: Optional[list[int]]) -> Optional[int]:
                    if matched is None or not exact_tag:
                        return None
                    p = _find_subseq_first(matched, exact_tag, start=0)
                    return p if p != -1 else None

                # stop if </answer> was hit
                if matched_seq is not None and _find_subseq_first(matched_seq, self.TAG_ANSWER_END, 0) != -1:
                    break
                if _strip_suffix_any(full_ids, self.V_ANSWER_END)[1] is not None:
                    break

                # stop segment on </reflect> is already guaranteed by boundary_needles;
                # but we continue the outer loop for next segment naturally.

                # tool_call hit?
                tool_hit = False
                tool_call_tag_start_idx = None

                off = _locate_exact(self.TAG_TOOL_CALL, matched_seq)
                if off is not None:
                    tool_call_tag_start_idx = (len(full_ids) - len(matched_seq)) + off
                    tool_hit = True
                else:
                    sfx = _strip_suffix_any(full_ids, self.V_TOOL_CALL)[1]
                    if sfx is not None:
                        off2 = _find_subseq_first(sfx, self.TAG_TOOL_CALL, 0)
                        if off2 != -1:
                            tool_call_tag_start_idx = (len(full_ids) - len(sfx)) + off2
                            tool_hit = True

                if tool_hit and tool_call_tag_start_idx is not None:
                    _append_ids(self.NEWLINE, gen_by_model=False)

                    base_pos = pos_ids[-1] + 1
                    base_prompt = list(full_ids)

                    tasks = [
                        asyncio.create_task(
                            self._gen_one_path(
                                base_prompt=base_prompt,
                                path_index=i + 1,
                                sampling_params=sampling_params,
                            )
                        )
                        for i in range(self.num_paths)
                    ]
                    paths = await asyncio.gather(*tasks)

                    path_spans: list[tuple[int, int]] = []
                    path_texts: list[str] = []

                    for path_body_ids, open_ids, close_ids, newline_ids in paths:
                        s = len(full_ids)
                        _append_ids(open_ids, gen_by_model=False)          # injected [pathi]
                        _append_ids(path_body_ids, gen_by_model=True)      # cleaned body
                        _append_ids(close_ids, gen_by_model=False)         # injected [/pathi]
                        _append_ids(newline_ids, gen_by_model=False)
                        e = len(full_ids)
                        path_spans.append((s, e))

                        # IMPORTANT: tool query should NOT include any [path*] tags
                        path_texts.append(self.tokenizer.decode(path_body_ids, skip_special_tokens=False))

                    longest = _align_paths_position_ids(path_spans, base_pos=base_pos)

                    _append_ids(self.TAG_TOOL_CALL_END, gen_by_model=True)
                    tool_call_end = len(full_ids)
                    _append_ids(self.NEWLINE, gen_by_model=False)

                    desired_after = base_pos + longest + 1
                    _shift_after(idx=tool_call_end, desired_start_pos=desired_after)

                    L_tc = tool_call_tag_start_idx
                    R_tc = tool_call_end - 1

                    for (l, r) in path_spans:
                        position_required_masks.append(
                            [
                                l,
                                (r - 1),
                                (L_tc + len(self.TAG_TOOL_CALL)),
                                (R_tc - (len(self.TAG_TOOL_CALL_END) + 1)),
                            ]
                        )

                    _append_ids(self.TAG_TOOL_RESP, gen_by_model=False)
                    _append_ids(self.NEWLINE, gen_by_model=False)

                    results_lines = []
                    for i, txt in enumerate(path_texts, start=1):
                        q = txt.strip()
                        ans = Search(q)
                        results_lines.append(f"[path{i}] {ans} [/path{i}]")
                    tool_body = "\n".join(results_lines) + "\n"
                    body_ids = self.tokenizer.encode(tool_body, add_special_tokens=False)

                    _append_ids(body_ids, gen_by_model=False)
                    _append_ids(self.TAG_TOOL_RESP_END, gen_by_model=False)

                    _scan_ops_until(len(full_ids))

                if len(full_ids) - init_len >= int(self.response_length) and self.response_length > 0:
                    break

        resp_ids = full_ids[init_len:]
        if len(resp_ids) > self.response_length and self.response_length > 0:
            resp_ids = resp_ids[: self.response_length ]
            resp_mask = resp_mask[: self.response_length ]
            pos_ids = pos_ids[: init_len + self.response_length ]
        
        if len(resp_ids) != len(resp_mask):
            raise ValueError("resp_ids and resp_mask length mismatch")
        if len(pos_ids) != len(prompt_ids) + len(resp_ids):
            raise ValueError("pos_ids length mismatch")
        
        final_text = self.tokenizer.decode(resp_ids, skip_special_tokens=False)
        final_text = final_text.replace("\n", "")
        print(f"Final generated response: {final_text}")

        out = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=resp_ids,
            response_mask=resp_mask,
            response_logprobs=None,
            routed_experts=None,
            multi_modal_data=None,
            reward_score=None,
            num_turns=iters,
            metrics=metrics,
            extra_fields={
                "position_ids_override": pos_ids,
                "position_required_masks": position_required_masks,
            },
        )
        return out
