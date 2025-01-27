from typing import Any, Literal, Union
from inspect_ai import Task, eval
from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

def mmlu_0_shot(cot: bool = False) -> Task:
    """
    Inspect Task implementation for MMLU, with 0-shot prompting.

    Args:
        cot (bool): Whether to use chain of thought
    """
    return Task(
        # (shuffle so that --limit draws from multiple subjects)
        dataset=get_mmlu_dataset(),
        solver=multiple_choice(cot=cot),
        scorer=choice(),
        config=GenerateConfig(temperature=0.5),
    )

def get_mmlu_dataset(
    shuffle: bool = False,
) -> Dataset:
    dataset = hf_dataset(
        path="cais/mmlu",
        name="high_school_physics", # Pick which subject, see https://huggingface.co/datasets/cais/mmlu
        split="dev", # change to "test" to get more samples
        sample_fields=record_to_sample,
        shuffle=True,
        seed=42,
    )
    return dataset


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["question"],
        choices=record["choices"],
        # converts 0 -> A, 1 -> B, etc.
        target=("ABCD"[record["answer"]]),
        metadata={"subject": record["subject"]},
    )

if __name__ == "__main__":
    models_to_compare = [ # https://inspect.ai-safety-institute.org.uk/models.html
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-3-5-haiku-latest",
    ]
    eval(mmlu_0_shot(), model=models_to_compare)