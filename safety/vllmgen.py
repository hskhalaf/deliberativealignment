import asyncio
import json
from safety.openai import OpenAI
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




def make_dataset():
    with open("prompts.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data]
    prompt_template = PromptTemplate()
    return prompt_template, prompts

async def fetch_response(model, question, prompt_template, timeout=300):

    n_generation = 10
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
        msg = response.choices[0].message
        if hasattr(msg, "reasoning_content"):
            reasoning_content = msg.reasoning_content
        else:
            reasoning_content = None
        content = msg.content
    except asyncio.TimeoutError:
        reasoning_content = None
        content = None
    
    return {
        "question": question,
        "response": content,
        "reasoning": reasoning_content
    }




async def main():
    models = client.models.list()
    model = models.data[0].id
    prompt_template, questions = make_dataset()
    batch_size = 64
    
    for i in range(0, len(questions), batch_size):
        batched_question = questions[i:i+batch_size]
        tasks = [fetch_response(model, q, prompt_template) for q in batched_question]
        batch_results = await asyncio.gather(*tasks)

        with open(f"safetyresults/{model.lower()}.jsonl", "a+") as f:
            for item in batch_results:
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
