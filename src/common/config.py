from dataclasses import dataclass


@dataclass
class GenerationConfig:
    num_generations: dict
    max_new_tokens: int
    max_length_for_gather: int
    max_generate_iterations: int
    temperature: float
    do_sample: bool
    use_diverse_sampling: bool
    diversity_penalty: float


@dataclass
class Batch:
    prompt: list
    answer: list
    context: list | None = None
