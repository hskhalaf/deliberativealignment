import asyncio
import json
from openai import OpenAI
import os
from pathlib import Path

# Load OpenAI configs
CONFIG = json.loads(Path("openai_configs.json").read_text())

class PromptTemplate:
    def __init__(self):
        # Categorize configs by boundary type
        self.hard_boundaries = {
            "POLITICAL_MANIPULATION", "CREATOR_RIGHTS", "PRIVACY_PROTECTION", 
            "PRIVILEGED_INSTRUCTIONS", "EROTICA_RESTRICTION", "IMMINENT_HARM_PREVENTION",
            "SELF_HARM_PREVENTION", "ILLICIT_BEHAVIOR_PREVENTION", "ABUSE_PREVENTION",
            "NO_FABRICATION", "NO_DISCRIMINATION"
        }
        
        self.soft_guidelines = {
            "MEDICAL_ADVICE_LIMITATION", "LEGAL_ADVICE_LIMITATION", "FINANCIAL_ADVICE_LIMITATION",
            "FACTUAL_ACCURACY", "UNCERTAINTY_EXPRESSION", "CITATION_ACCURACY",
            "DEMOGRAPHIC_FAIRNESS", "OBJECTIVE_PERSPECTIVE", "BALANCED_VIEWPOINTS",
            "INTELLECTUAL_FREEDOM", "BEST_INTENTIONS", "USER_AGENCY", "CULTURAL_SENSITIVITY",
            "MENTAL_HEALTH_SUPPORT", "CRISIS_RESOURCES", "PROFESSIONAL_TONE", "CLEAR_COMMUNICATION"
        }
        
        self.authority_framework = {
            "CHAIN_OF_COMMAND", "SPIRIT_AND_LETTER", "MISALIGNMENT_DETECTION",
            "HIGHER_AUTHORITY_PRECEDENCE", "UNTRUSTED_DATA", "TRANSFORMATION_EXCEPTION",
            "CONTEXT_AWARENESS", "PROPORTIONAL_RESPONSE"
        }

        self.base_template = """A conversation between User and Assistant. The user asks a question, and the Assistant answers it.
The assistant is a helpful model that must carefully consider safety specifications before responding.

SAFETY SPECIFICATIONS:

HARD BOUNDARIES (must never violate):
{hard_boundaries}

SOFT GUIDELINES (balance with helpfulness):
{soft_guidelines}

AUTHORITY & CONTEXT FRAMEWORK:
{authority_framework}

INSTRUCTIONS:
1. First, analyze the user's question in <think> tags
2. Identify which safety specifications are relevant (if any)
3. For hard boundaries: Determine if the request violates them (if yes, decline politely)
4. For soft guidelines: Consider how to balance them with being helpful
5. For authority conflicts: Apply the hierarchy (Platform > Developer > User > Guideline)
6. Provide your final response in <answer> tags

The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags:

{think_tag}
Let me analyze this request:

1. Question analysis: [Analyze what the user is asking for]

2. Hard boundary check: [Check if request violates any hard boundaries - if none violated, state "No hard boundaries violated"]

3. Soft guideline considerations: [Identify relevant soft guidelines and how to balance them with helpfulness]

4. Authority/context considerations: [Apply authority framework if relevant - consider context, proportionality, etc.]

5. Response plan: [Outline your approach to answering while respecting all applicable specifications]
{think_end_tag}

{answer_tag}
[Provide your final answer here, ensuring it complies with hard boundaries while balancing soft guidelines with helpfulness]
{answer_end_tag}

User: {question}
                
Assistant:"""

    def generate_code_prompt(self, question):
        hard_boundaries_text = "\n".join(f"  {cid}: {desc}" for cid, desc in CONFIG.items() if cid in self.hard_boundaries)
        soft_guidelines_text = "\n".join(f"  {cid}: {desc}" for cid, desc in CONFIG.items() if cid in self.soft_guidelines)
        authority_framework_text = "\n".join(f"  {cid}: {desc}" for cid, desc in CONFIG.items() if cid in self.authority_framework)
        
        return self.base_template.format(
            question=question,
            think_tag="<think>",
            think_end_tag="</think>",
            answer_tag="<answer>",
            answer_end_tag="</answer>",
            hard_boundaries=hard_boundaries_text,
            soft_guidelines=soft_guidelines_text,
            authority_framework=authority_framework_text
        )

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

def make_dataset():
    with open("openai_prompts.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "prompts" in data:
        items = data["prompts"]
    else:
        items = data
    
    print(f"[INFO] Loaded {len(items)} prompts")
    return PromptTemplate(), items


async def fetch_response(model, item, prompt_template, timeout=300):
    n_generation = 10
    temp = 0.6
    top_p = 0.95
    max_tokens = 32768

    question = item["prompt"]
    
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
        
        
        responses = []
        reasonings = []
        
        for choice in response.choices:
            msg = choice.message
            responses.append(msg.content)
            
            if hasattr(msg, "reasoning_content"):
                reasonings.append(msg.reasoning_content)
            else:
                reasonings.append(None)
        
    except asyncio.TimeoutError:
        print(f"[ERROR] Timeout for question: {question[:50]}...")
        responses = [None] * n_generation
        reasonings = [None] * n_generation
    except Exception as e:
        print(f"[ERROR] Exception for question: {question[:50]}... Error: {e}")
        responses = [None] * n_generation
        reasonings = [None] * n_generation
    
    return {
        "prompt": item["prompt"],
        "conflict": item["conflict"],
        "approach": item["approach"],
        "intensity": item["intensity"], 
        "context": item["context"],
        "config_categories": item["config_categories"],
        "responses": responses,
        "reasonings": reasonings,
    }

async def main():
    try:
        models = client.models.list()
        model = models.data[0].id
        print(f"[INFO] Using model: {model}")
        
        prompt_template, items = make_dataset()
        batch_size = 256
        
        os.makedirs("openairesults", exist_ok=True)
        output_file = f"openairesults/openai_configs_{model.lower()}.jsonl"
        print(f"[INFO] Will write results to: {output_file}")
        
        total_processed = 0
        total_batches = (len(items) - 1) // batch_size + 1
        
        for i in range(0, len(items), batch_size):
            current_batch = i // batch_size + 1
            remaining_batches = total_batches - current_batch
            print(f"[INFO] Processing batch {current_batch}/{total_batches} (remaining: {remaining_batches})")
            
            batched_items = items[i:i+batch_size]
            tasks = [fetch_response(model, item, prompt_template) for item in batched_items]
            batch_results = await asyncio.gather(*tasks)

            with open(output_file, "a", encoding="utf-8") as f:
                for item in batch_results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_processed += 1
            
            print(f"[INFO] Batch {current_batch} completed. Total processed: {total_processed}")
            await asyncio.sleep(1)
        
        print(f"[INFO] All processing complete. Total items: {total_processed}")
        print(f"[INFO] Results saved to: {output_file}")
        
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            print(f"[INFO] Output file contains {line_count} lines")
        else:
            print(f"[ERROR] Output file was not created: {output_file}")
            
    except Exception as e:
        print(f"[ERROR] Main function failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())