import asyncio
import json
from datasets import load_dataset
from openai import OpenAI
from vllm import LLM, SamplingParams


class PromptTemplate:
    def __init__(self):
        self.base_template = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
                                The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
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
            reasoning=""" Provide reasoning here
                        1. Analyze the programming requirements
                        2. Consider edge cases and constraints
                        3. Plan the algorithm structure
                        4. Think about time and space complexity
                        5. Consider test cases""",
            think_end_tag="</think>",
            answer_tag="<answer>",
            solution="[Provide the code solution here]",
            answer_end_tag="</answer>"
        )

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(api_key=openai_api_key,
        base_url=openai_api_base)


def make_dataset(dataset_name='verifiable-coding-problems-python-10k'):

    ds = load_dataset(f"open-r1/{dataset_name}")
    ids = []
    questions = []
    answers = []
    for entry in ds["train"]:
        if entry.get("verification_info") and entry.get("verification_info").get("language") == "python":
            questions.append(entry["problem"])
            answers.append({
                "gold_standard": entry["gold_standard_solution"],
                "test_cases": entry["verification_info"]["test_cases"]
            })
            ids.append(entry['in_source_id'])
    
    prompt_template = PromptTemplate()

    return prompt_template, questions, answers, ids

async def fetch_response(model, question, expected_output, prompt_template, temp=0.2, top_p=0.95, timeout=300):

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
        "expected_output": expected_output,
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

    prompt_template, questions, answers, ids = make_dataset()
    # sampling_params = SamplingParams(temperature=temp, top_p=top_p)

    # llm = LLM(f"deepseek-ai/DeepSeek-R1-Distill-{model}", tensor_parallel_size=n_gpus, trust_remote_code=True)
    
    results = []
    batch_size = 64
    for i in range(0, len(questions), batch_size):
        print(i)
        batched_question = questions[i:i+batch_size]
        batched_answer = answers[i:i+batch_size]
        tasks = [fetch_response(model, q, exp, prompt_template) for q, exp in zip(batched_question, batched_answer)]
        # results = run_llms(llm, sampling_params, formatted_question)
        batch_results = await asyncio.gather(*tasks)
        # results.extend(batch_results)
        print(batch_results[0]['response'])

        with open(f"results/{model.lower()}.jsonl", "a+") as f:
            for item in batch_results:
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
