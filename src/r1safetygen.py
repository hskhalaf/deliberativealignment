import asyncio
import json
from datasets import load_dataset
from openai import OpenAI
from vllm import LLM, SamplingParams
import pandas as pd
from datasets import Dataset
import requests
import gzip
import shutil
import os

class PromptTemplate:
    def __init__(self):
        self.base_template = """A conversation between User and Assistant. The user asks a question, and the Assistant answers it.
                                The assistant is a helpful model that first thinks about the reasoning process in the mind and then provides the user with the answer.
                                The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.
                                {think_tag}
                                {reasoning}
                                {think_end_tag}
                                {answer_tag}
                                {solution}
                                {answer_end_tag}
                                User: {question} \n
                                Assistant: 
                                """
    def generate_code_prompt(self, question):
        return self.base_template.format(
            question=question,
            think_tag="<think>",
            reasoning=""" Provide reasoning here""",
            think_end_tag="</think>",
            answer_tag="<answer>",
            solution="[Provide the answer here]",
            answer_end_tag="</answer>"
        )

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(api_key=openai_api_key,
        base_url=openai_api_base)


def download_and_decompress(url, output_file):
    compressed_file = output_file + ".gz"
    response = requests.get(url)
    if response.status_code == 200:
        with open(compressed_file, "wb") as f:
            f.write(response.content)
        with gzip.open(compressed_file, "rb") as gz, open(output_file, "wb") as out:
            shutil.copyfileobj(gz, out)
    else:
        raise Exception(f"Failed to download {compressed_file}: HTTP {response.status_code}")

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def prompt(text):
    last_assistant = text.rfind("Assistant:")
    return text[:last_assistant].strip()

def process_default(dataset):
    data_list = dataset.to_list()
    df = pd.DataFrame({
        "prompt": [prompt(text["chosen"]) for text in data_list],
    })
    return Dataset.from_pandas(df)

def make_dataset():
    base_url = "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/harmless-base/"
    for file in ["train.jsonl.gz", "test.jsonl.gz"]:
        download_and_decompress(base_url + file, file.replace(".gz", ""))
    data_harmless = Dataset.from_list(load_jsonl("train.jsonl")).select(range(20000))
    data_harmless = process_default(data_harmless)

    questions = data_harmless["prompt"]
    prompt_template = PromptTemplate()

    return prompt_template, questions

async def fetch_response(model, question, prompt_template, temp=0.2, top_p=0.95, timeout=300):

    n_generation = 1
    temp = 0.6
    top_p = 0.95
    max_tokens = 32768

    formatted_prompt = prompt_template.generate_code_prompt(question)
    messages = [{"role": "user", "content": formatted_prompt}]
    loop = asyncio.get_running_loop()
    try:
        response = await asyncio.wait_for(loop.run_in_executor(None, lambda: client.chat.completions.create(
            model=model, 
            messages=messages, 
            n=n_generation,
            temperature=temp,
            top_p=top_p, 
            max_tokens=max_tokens,
        )), timeout)
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
    except asyncio.TimeoutError:
        reasoning_content = None
        content = None
    
    return {
        "question": question,
        "response": content,
        "reasoning": reasoning_content
    }

# deprecated
def run_llms(model, sampling_params, batch_questions):
    outputs = model.generate(batch_questions, sampling_params)
    # for output in outputs:
        # response = output.outputs[0].
    # response = outputs

async def main():
    
    models = client.models.list()
    model = models.data[0].id

    dataset = 'verifiable-coding-problems-python-10k'
    # model = 'Qwen-14B'
    n_gpus = 4

    prompt_template, questions = make_dataset()
    # sampling_params = SamplingParams(temperature=temp, top_p=top_p)

    # llm = LLM(f"deepseek-ai/DeepSeek-R1-Distill-{model}", tensor_parallel_size=n_gpus, trust_remote_code=True)
    
    results = []
    batch_size = 64
    file_path = f"safetyresults/{model.lower()}.jsonl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    for i in range(0, len(questions), batch_size):
        print(i)
        batched_question = questions[i:i+batch_size]
        tasks = [fetch_response(model, q, prompt_template) for q in batched_question]
        # results = run_llms(llm, sampling_params, formatted_question)
        batch_results = await asyncio.gather(*tasks)
        # results.extend(batch_results)
        print(batch_results[0]['response'])

        with open(f"safetyresults/{model.lower()}.jsonl", "a+") as f:
            for item in batch_results:
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
