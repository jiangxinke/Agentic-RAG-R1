# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import glob
import pandas as pd

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Prompt configuration（❗一字不改）
# -----------------------------------------------------------------------------
DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."

DEFAULT_USER_CONTENT_PREFIX = """
<system instruction>
When the user asks a question, the assistant should actively solve it. The assistant may think, tool_call, reflect, and then produce a final answer. Use the following structured tags to organize reasoning and tool_call steps. Precise formatting matters — follow the rules below.

Tags (semantic roles)
1. <think> ... </think>
   - Use to record the assistant's internal think, step-by-step analysis, or intermediate thoughts that explain how the assistant reached a conclusion.
2. <tool_call> ... </tool_call>
   - Use when the assistant must perform external or uncertain information retrieval.
   - The content must follow this exact format:
     "<tool_call> keyword_1 keyword_2 ... </tool_call>"
   - After sending the tool_call tag, the system/tool will return results wrapped in an "<tool_response> ... </tool_response>" block.
3. <reflect> ... </reflect>
   - Use when summarizing or consolidating previous reasoning or conclusions,
     OR when previous think or conclusions need correction or revision.
   - Clearly state whether this reflect is a summary or a correction, and explain what changed and why if applicable.
4. <answer> ... </answer>
   - Provide the final answer to the user's question.
   - This tag must appear exactly once and must be placed at the very end of the response.

Strict rules and formatting constraints
1. Only the <answer> tag is required to appear exactly once — and it must appear only at the end of the assistant's response.
2. All other tags (<think>, <tool_call>, <reflect>) may appear multiple times in any order, as needed.
3. Maintain exact tag spelling and angle-bracket punctuation. Tags are case-sensitive.
4. If a <tool_call> tag is used, expect a follow-up "<tool_response> ... </tool_response>" from the system and incorporate that tool_response into subsequent think or the final answer.
5. Keep think clear and focused — long internal chains of thought may be split across multiple <think> blocks if appropriate.
6. ALL assistant output MUST be enclosed within one of the defined tags.
7. NO plain text or symbols are allowed outside of tags.
8. Tags MUST NOT be nested. Each piece of content must belong to exactly one top-level tag.

Behavioral guidance
- Be concise, truthful, and helpful.
- When you reflect, explicitly state what you changed and why.
- The final <answer> should be a clear, stand-alone response that a user could read without needing to see the intermediate tags (though including a brief reflect of the think is allowed if it helps clarity).
- Avoid leaking internal-only control signals or non-human-readable tokens outside the structured tags.

</system instruction>

<query>
"""


def process_single_row(row, current_split_name, row_index, data_source_tagged="searchR1_asearcher"):
    question = row.get("question", "")
    answer = row.get("answer", "")

    user_content = (
        DEFAULT_USER_CONTENT_PREFIX.rstrip("\n")
        + question
        + "\n</query>"
    )

    prompt = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": user_content},
    ]

    ground_truth = [answer] if isinstance(answer, str) and answer else []

    # data_source_tagged = "searchR1_asearcher"

    reward_model = {
        "ground_truth": {"target": ground_truth},
        "style": "rule",
    }

    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "data_source": data_source_tagged,
                "ground_truth": {"target": ground_truth},
                "question": question,
            }
        }
    }

    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": "fact-reasoning",
            "reward_model": reward_model,
            "extra_info": extra_info,
            "metadata": None,
        }
    )

def process_split(parquet_files, split_name, output_dir):
    l = len("/data2/gjr/workshop/r1/data/ASearcher/test_")
    r = len(".parquet")
    print(f"Processing {split_name} split with files: {parquet_files}")
    dfs = []
    src_names = []
    for p in parquet_files:
        dfs.append(pd.read_parquet(p))
        src_names.append((p[l:])[:-r])
        
    df_raw = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged {split_name} rows: {len(df_raw)}")
    src_col = []
    for p, df in zip(src_names, dfs):
        src_col.extend([p] * len(df))

    df_raw = df_raw.assign(src=src_col)

    df_processed = df_raw.apply(
        lambda row: process_single_row(
            row,
            current_split_name=split_name,
            row_index=row.name,
            data_source_tagged=row["src"],
        ),
        axis=1,
    )
    
    out_path = os.path.join(output_dir, f"{split_name}.parquet")
    df_processed.to_parquet(out_path, index=False)
    logger.info(f"Saved {split_name} parquet to {out_path}")


def main():
    output_dir = os.path.expanduser(args.local_dir)
    os.makedirs(output_dir, exist_ok=True)

    # -------- train --------
    train_files = sorted(glob.glob(os.path.join(args.asearcher_dir, "train*.parquet")))
    if train_files:
        process_split(train_files, "train", output_dir)
    else:
        logger.warning("No train parquet files found")

    # -------- test --------
    test_files = sorted(glob.glob(os.path.join(args.asearcher_dir, "test_*.parquet")))
    if test_files:
        process_split(test_files, "test", output_dir)
    else:
        logger.warning("No test parquet files found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SearchR1 train/test parquet from ASearcher data."
    )

    parser.add_argument(
        "--asearcher_dir",
        default="/data2/gjr/workshop/r1/data/ASearcher",
        help="Directory containing train*.parquet and test_*.parquet",
    )
    parser.add_argument(
        "--local_dir",
        default="/data2/gjr/workshop/r1/data/searchR1_processed_direct",
        help="Output directory",
    )

    args = parser.parse_args()
    main()
