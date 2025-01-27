## Installation

1. Clone this repository

2. (optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set your AI API keys:
```bash
export OPENAI_API_KEY=api-key-here # Ask us for one
export ANTHROPIC_API_KEY=api-key-here # Ask us for one
```

## Example files

### chat.py

Ask a single question to an AI through the CLI. You can modify the system message to get more control.

```bash
python chat.py "your question here"
```
This script can be usefull for testing different models on different questions with different system messages.

### dataset_evaluation.py

Meassure how good an AI is at answering multiple choice questions from a dataset.

To run the evaluation,
```bash
python dataset_evaluation.py
```
this will create a `/logs` folder with the results.

To view the results,
```bash
inspect view
```
then go to `http://127.0.0.1:7575` to view the results.

This script uses the [MMLU](https://huggingface.co/datasets/cais/mmlu), can you create your own dataset?

### computer_use.py

Meassure how good an AI is at clicking on UI elements on a computer.

To run the evaluation,
```bash
python computer_use.py
```
this will create a `/logs` folder with the results.

To view the results,
```bash
inspect view
```
then go to `http://127.0.0.1:7575` to view the results.

## Pointers

### Inspect-ai

Inspect-ai is a framework for large language model evaluations.

Go to `https://inspect.ai-safety-institute.org.uk/` to learn more.

### Openrouter

In Openrouter you can chat with many different AI models from many different providers. It is like ChatGPT, but you can pick models from OpenAI, Anthropic, or open-source models.

Go to `https://openrouter.ai/chat` and click `Add model`.

If you want to use it with OpenAI/Anthropic, go to `https://openrouter.ai/settings/integrations` to set your API key.