from typing import List, Tuple, Union, Optional, Dict, Any
import torch


DEFAULT_ACTION_CONFIG = {
    "think": ("<think>", "</think>"),
    "reflect": ("<reflect>", "</reflect>"),
    "tool_call": ("<tool_call>", "</tool_call>"),
    "answer": ("<answer>", "</answer>")
}

class TokenRecognizer:
    def __init__(self, tokenizer=None):
        """
        Initialize the TokenRecognizer.
        
        Args:
            tokenizer: A huggingface tokenizer (optional). If provided, can resolve string tokens to IDs.
        """
        self.tokenizer = tokenizer

    def _resolve_id(self, token: Union[int, str]) -> int:
        if isinstance(token, int):
            return token
        if self.tokenizer is None:
            raise ValueError(f"Tokenizer not provided, cannot resolve string token: {token}")
        
        # Try to encode
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 0:
            # Maybe it is a special token that isn't in normal vocab?
            # Try convert_tokens_to_ids
            tid = self.tokenizer.convert_tokens_to_ids(token)
            if tid == self.tokenizer.unk_token_id and token != self.tokenizer.unk_token:
                 raise ValueError(f"Token '{token}' resolves to UNK or empty")
            return tid
        elif len(ids) > 1:
            # Check if it's a single special token
            tid = self.tokenizer.convert_tokens_to_ids(token)
            if tid != self.tokenizer.unk_token_id:
                return tid
            # If it really encodes to multiple tokens, we only support single token matching for now
            # unless we implement subsequence matching.
            # Assuming the user asks for "special tokens" which are usually single IDs.
            raise ValueError(f"Token '{token}' encodes to multiple IDs {ids}, single ID expected.")
        return ids[0]

    def find_positions(
        self, 
        input_ids: Union[List[int], torch.Tensor], 
        token: Union[int, str]
    ) -> List[int]:
        """
        Find all positions of a specific token.
        
        Args:
            input_ids: List of token IDs or 1D Tensor.
            token: Token ID (int) or string (if tokenizer is present).
            
        Returns:
            List of indices where the token appears.
        """
        target_id = self._resolve_id(token)
        
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() > 1:
                raise ValueError("input_ids must be 1D")
            input_ids = input_ids.tolist()
            
        return [i for i, x in enumerate(input_ids) if x == target_id]

    def find_intervals(
        self,
        input_ids: Union[List[int], torch.Tensor],
        start_token: Union[int, str],
        end_token: Union[int, str],
        include_boundaries: bool = True,
        allow_nested: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Find intervals between start_token and end_token.
        Returns list of (start, end) tuples using Python slice semantics [start, end).
        
        Args:
            input_ids: List of token IDs or 1D Tensor.
            start_token: Start tag/token.
            end_token: End tag/token.
            include_boundaries: If True, interval includes the start and end tokens.
                                If False, interval is strictly between them.
            allow_nested: (Not implemented, simplified to flat matching). 
                          Current logic: A new start token resets the current interval start.
                          
        Returns:
            List of (start_index, end_index).
        """
        start_id = self._resolve_id(start_token)
        end_id = self._resolve_id(end_token)
        
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() > 1:
                raise ValueError("input_ids must be 1D")
            input_ids = input_ids.tolist()
            
        intervals = []
        current_start_idx = -1
        
        for i, tid in enumerate(input_ids):
            if tid == start_id:
                # If we encounter a start token, we mark it.
                # If we were already waiting for an end, this effectively "restarts" the block
                # (handling cases like: <think> ... oops <think> ... </think>)
                current_start_idx = i
            elif tid == end_id:
                if current_start_idx != -1:
                    # Found a pair
                    if include_boundaries:
                        # From start_token up to and including end_token
                        # Python slice: [start, end + 1)
                        intervals.append((current_start_idx, i + 1))
                    else:
                        # Strictly between
                        # Python slice: [start + 1, end)
                        # Ensure we don't return invalid range if they are adjacent
                        s = current_start_idx + 1
                        e = i
                        if e >= s:
                            intervals.append((s, e))
                        else:
                            # Empty interval
                            intervals.append((s, s))
                    
                    # Reset
                    current_start_idx = -1
                    
        return intervals

    def segment_batch(
        self,
        input_ids_batch: Union[List[List[int]], List[int], torch.Tensor],
        action_config: Optional[Dict[str, Tuple[Union[int, str], Union[int, str]]]] = None,
        include_boundaries: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Segment a batch of input_ids based on defined actions (start/end tokens).
        
        Args:
            input_ids_batch: Batch of input IDs (2D list/tensor) or single sequence (1D list/tensor).
            action_config: Dictionary mapping action names to (start_token, end_token) tuples.
                           Defaults to DEFAULT_ACTION_CONFIG ({'reflect': ..., 'tool_call': ...}).
            include_boundaries: Whether to include the start/end tokens in the segments.
            
        Returns:
            List of lists of dictionaries. 
            Outer list corresponds to batch items.
            Inner list corresponds to the ordered sequence of actions found.
            Each dictionary contains:
                - step: Sequence number of the action
                - action: Name of the action
                - start_idx: Start index in the input_ids
                - end_idx: End index in the input_ids
                - token_ids: The list of token IDs for this segment
                - text: Decoded string content (if tokenizer is available)
        """
        if action_config is None:
            action_config = DEFAULT_ACTION_CONFIG

        # Normalize input to List[List[int]]
        if isinstance(input_ids_batch, torch.Tensor):
            if input_ids_batch.dim() == 1:
                input_ids_batch = input_ids_batch.unsqueeze(0).tolist()
            else:
                input_ids_batch = input_ids_batch.tolist()
        elif isinstance(input_ids_batch, list):
            if len(input_ids_batch) > 0 and isinstance(input_ids_batch[0], int):
                input_ids_batch = [input_ids_batch]
        
        batch_results = []
        
        for seq_ids in input_ids_batch:
            found_actions = []
            
            # Search for each action type
            for action_name, (start_token, end_token) in action_config.items():
                intervals = self.find_intervals(
                    seq_ids, 
                    start_token, 
                    end_token, 
                    include_boundaries=include_boundaries
                )
                
                for start, end in intervals:
                    # Extract content
                    content_ids = seq_ids[start:end]
                    content_text = None
                    if self.tokenizer:
                        content_text = self.tokenizer.decode(content_ids)
                        
                    found_actions.append({
                        "action": action_name,
                        "start_idx": start,
                        "end_idx": end,
                        "token_ids": content_ids,
                        "text": content_text
                    })
            
            # Sort actions by start index to establish the sequence "step 1, step 2..."
            found_actions.sort(key=lambda x: x['start_idx'])
            
            # Add step numbers
            for i, item in enumerate(found_actions):
                item['step'] = i + 1  # 1-based index as requested ("step 1...")
                
            batch_results.append(found_actions)
            
        return batch_results

if __name__ == '__main__':
    # Mock Tokenizer for demonstration
    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                "<pad>": 0,
                "<reflect>": 10,
                "</reflect>": 11,
                "<tool_call>": 20,
                "</tool_call>": 21,
                "<think>": 30,
                "</think>": 31,
                "<answer>": 40,
                "</answer>": 41,
                "hello": 100,
                "world": 101
            }
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
            self.unk_token_id = 999
            self.unk_token = "<unk>"
            
        def encode(self, text, add_special_tokens=False):
            if text in self.vocab:
                return [self.vocab[text]]
            return [self.unk_token_id]
            
        def convert_tokens_to_ids(self, token):
            return self.vocab.get(token, self.unk_token_id)
            
        def decode(self, ids):
            return " ".join([self.ids_to_tokens.get(i, "<unk>") for i in ids])

    # Initialize Recognizer
    tokenizer = MockTokenizer()
    recognizer = TokenRecognizer(tokenizer)
    
    # Define input batch
    # Sequence 1: <reflect> hello world </reflect> ... <tool_call> hello </tool_call> ... <answer> world </answer>
    # IDs: 10, 100, 101, 11, 0, 0, 20, 100, 21, 0, 40, 101, 41
    input_ids_1 = [10, 100, 101, 11, 0, 0, 20, 100, 21, 0, 40, 101, 41]
    
    # Sequence 2: <think> hello </think> ... <tool_call> world </tool_call>
    # IDs: 30, 100, 31, 0, 20, 101, 21
    input_ids_2 = [30, 100, 31, 0, 20, 101, 21]
    
    batch_input = [input_ids_1, input_ids_2]
    
    # Use default action config
    
    print("Processing batch...")
    results = recognizer.segment_batch(batch_input)
    from rich import print
    print("=" * 50)
    print(results)
    print("=" * 50)
    
    for idx, steps in enumerate(results):
        print(f"\n--- Sequence {idx+1} ---")
        if not steps:
            print("No actions found.")
            continue
        for step in steps:
            print(f"Step {step['step']} [{step['action']}]:")
            print(f"  Range: {step['start_idx']} - {step['end_idx']}")
            print(f"  Token IDs: {step['token_ids']}")
            print(f"  Text: {step['text']}")
