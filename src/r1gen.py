import asyncio
import json
from datasets import load_dataset
from openai import OpenAI

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

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

ds = load_dataset("open-r1/verifiable-coding-problems-python-10k")
questions = []
expected_outputs = []
for entry in ds["train"]:
    if entry.get("verification_info") and entry.get("verification_info").get("language") == "python":
        questions.append(entry["problem"])
        expected_outputs.append({
            "gold_standard": entry["gold_standard_solution"],
            "test_cases": entry["verification_info"]["test_cases"]
        })

prompt_template = PromptTemplate()

async def fetch_response(question, expected_output, timeout=300):
    formatted_prompt = prompt_template.generate_code_prompt(question)
    messages = [{"role": "user", "content": formatted_prompt}]
    loop = asyncio.get_running_loop()
    try:
        response = await asyncio.wait_for(loop.run_in_executor(None, lambda: client.chat.completions.create(
            model=model, messages=messages, temperature=0.2
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

async def main():
    results = []
    batch_size = 150
    total = 1200
    for i in range(0, total, batch_size):
        batch_questions = questions[i:i+batch_size]
        batch_expected = expected_outputs[i:i+batch_size]
        tasks = [fetch_response(q, exp) for q, exp in zip(batch_questions, batch_expected)]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    with open("results_qwen14b.jsonl", "a") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

asyncio.run(main())
