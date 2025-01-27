import os
import json
import random
import math

from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import solver, chain, generate, system_message, chain_of_thought, TaskState
from inspect_ai.scorer import (
    scorer,
    Score,
    Target,
    accuracy,
    mean,
    stderr,
)
from inspect_ai.model import ChatMessageUser, ContentText, ContentImage

@scorer(
    metrics=[{
        "hit": [accuracy(), stderr()],
        "distance": [mean(), stderr()],
        "hit_area_weighted": [mean(), stderr()]
    }]
)
def click_scorer():
    """
    We expect the model to output: "x=NNN, y=NNN"
    We'll parse that and see if it matches the target coordinates.
    Then we compute distance and area-weighted scores.
    """

    async def score_fn(state: TaskState, target: Target):
        """Compare the model's click to the target coordinates."""
        # Parse target coordinates from JSON string
        try:
            ans = state.output.completion.split("<")[1].split(">")[0]
            x, y = ans.split(",")
            click_x = int(x.strip())
            click_y = int(y.strip())
        except Exception as exc:
            click_x, click_y = 0, 0
        
        target_dict = json.loads(target.text)
        target_x = target_dict["x"]
        target_y = target_dict["y"]
        width = target_dict["width"]
        height = target_dict["height"]
        area = width * height

        # Check if click is inside the bounding box with tolerance
        tolerance = 10
        hit = 1 if (target_x - tolerance <= click_x <= target_x + width + tolerance and 
                   target_y - tolerance <= click_y <= target_y + height + tolerance) else 0

        # Calculate distance to closest boundary
        dx = max(target_x - click_x, 0, click_x - (target_x + width))
        dy = max(target_y - click_y, 0, click_y - (target_y + height))
        distance = (dx ** 2 + dy ** 2) ** 0.5
        log_distance = math.log(distance + 1)  # add 1 to handle distance=0 case

        area_score = hit * 1000 / math.log(area + 1)

        return Score(
            value={
                "hit": hit,
                "distance": log_distance,
                "hit_area_weighted": area_score
            },
            answer=f"x={click_x}, y={click_y}",
        )

    return score_fn

@solver
def coordinate_locator():
    """
    The reference examples will now be included in each sample's input message,
    before the target element description.
    """
    return chain(
        system_message(
"""You are a coordinate-locator-bot.

You are given a screenshot from a computer screen and descriptions of UI elements.
First you will see some reference examples with their coordinates for scale.
Then you will see the target element to locate.
Your task is to output the pixel coordinates of the target element.

Motivate your reasoning in one sentence and then output the coordinates inside angle brackets at the end. Like this:
<100,200>
"""
        ),
        chain_of_thought(),
        generate()
    )

def locate_ui_elements(
    few_shot_num: int = 2,
    max_elements_per_file: int = 5,
) -> Task:
    """
    - Reads all annotation JSON from dataset/annotations/.
    - For each image, uses few_shot_num elements as reference examples
    - For the actual evaluation, uses the remaining elements
    - Returns a Task that uses the coordinate_locator solver and click_scorer.
    """
    annotations_dir = "dataset/annotations"
    all_samples = []

    random_seed = 42
    rng = random.Random(random_seed)

    for fname in sorted(os.listdir(annotations_dir)):
        if not (fname.startswith("annotation_") and fname.endswith(".json")):
            continue

        path = os.path.join(annotations_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)

        elements = data.get("elements", [])
        if not elements or len(elements) <= few_shot_num:
            continue

        # Shuffle elements for this file
        rng.shuffle(elements)
        
        # Split into reference and evaluation elements
        reference_elements = elements[:few_shot_num]
        num_eval_elements = min(max_elements_per_file, len(elements) - few_shot_num)
        eval_elements = elements[few_shot_num:few_shot_num + num_eval_elements]

        # Format reference examples for this image
        reference_text = "Reference examples:\n"
        for ref in reference_elements:
            desc = ref["description"].strip()
            box = ref["bounding_box"]
            center_x = box["x"] + box["width"] // 2
            center_y = box["y"] + box["height"] // 2
            reference_text += f"- Element '{desc}' is at x={center_x}, y={center_y}\n"
        reference_text += "\nNow locate this element:\n"

        # Create samples for evaluation elements
        screenshot_path = data.get("screenshot_path", "")
        
        for elem in eval_elements:
            desc = elem["description"].strip()
            if not desc:
                continue

            bbox = elem["bounding_box"]
            target_coords = json.dumps({
                "x": bbox["x"],
                "y": bbox["y"],
                "width": bbox["width"],
                "height": bbox["height"]
            })

            input_msg = []
            if screenshot_path:
                # Compress image if needed
                try:
                    input_msg.append(ChatMessageUser(
                        content=[
                            ContentText(text=reference_text + desc),
                            ContentImage(image=screenshot_path)
                        ]
                    ))
                except Exception as e:
                    print(f"Warning: Could not process image {screenshot_path}: {e}")
                    continue
            else:
                input_msg.append(ChatMessageUser(content=reference_text + desc))

            sample = Sample(
                input=input_msg,
                target=target_coords,
                metadata={
                    "url": data["url"],
                    "screenshot": screenshot_path,
                    "annotation_id": elem["id"]
                }
            )
            all_samples.append(sample)

    dataset = MemoryDataset(all_samples)
    return Task(
        dataset=dataset,
        solver=coordinate_locator(),
        scorer=click_scorer(),
    )

def main():
    models = [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet-latest",
    ]

    # Build the task
    the_task = locate_ui_elements(
        few_shot_num=2,
        max_elements_per_file=1,
    )
    eval(the_task, model=models)

if __name__ == "__main__":
    main()