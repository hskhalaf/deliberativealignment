import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    GenerationConfig,
    TextIteratorStreamer,
)
from tqdm.asyncio import tqdm

# ──────────────────────────────────────────────────────────────────────────
# 1.  Prompt template
# ──────────────────────────────────────────────────────────────────────────
class PromptTemplate:
    """Matches the structure you used with vLLM / OpenAI."""

    _template = (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant answers it.\n\n"
        "The assistant is a helpful model that first thinks about the reasoning "
        "process in the mind and then provides the user with the answer.\n"
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, i.e.\n"
        "{think_tag}\n"
        "{reasoning}\n"
        "{think_end_tag}\n"
        "{answer_tag}\n"
        "{solution}\n"
        "{answer_end_tag}\n\n"
        "User: {question}\n"
        "Assistant:"
    )

    def generate(self, question: str) -> str:
        return self._template.format(
            question=question.strip(),
            think_tag="<think>",
            reasoning="Provide reasoning here",
            think_end_tag="</think>",
            answer_tag="<answer>",
            solution="[Provide the answer here]",
            answer_end_tag="</answer>",
        )


# ──────────────────────────────────────────────────────────────────────────
# 2.  Thinking‑token budget processor
# ──────────────────────────────────────────────────────────────────────────
class ThinkingTokenBudgetProcessor(LogitsProcessor):
    """
    Alternative approach: Track how many generation steps we've taken
    instead of comparing absolute token positions
    """

    def __init__(self, tokenizer, max_thinking_tokens: int):
        super().__init__()
        self.tok = tokenizer
        self.max_tokens = max_thinking_tokens

        self.think_end = tokenizer.encode("</think>", add_special_tokens=False)
        self.newline = tokenizer.encode("\n", add_special_tokens=False)[0]

        # Track generation steps instead of absolute positions
        self.generation_step = 0
        self.in_think = True  # Start in think mode since <think> is in prompt
        self.NINF = float("-inf")
        

    def _endswith(self, ids: List[int], suffix: List[int]) -> bool:
        return len(ids) >= len(suffix) and ids[-len(suffix) :] == suffix

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Each call to this method represents one generation step
        self.generation_step += 1
        ids = input_ids[0].tolist()
        
        if self.in_think:
            # print(f"DEBUG: Generation step {self.generation_step}/{self.max_tokens}")
            
            # Check if we naturally hit </think>
            if self._endswith(ids, self.think_end):
                # print(f"DEBUG: Natural </think> found at step {self.generation_step}")
                self.in_think = False
                return scores
            
            # Check if we've hit our budget
            if self.generation_step >= self.max_tokens:
                # print(f"DEBUG: Hit budget at step {self.generation_step}, forcing </think>")
                scores[0].fill_(self.NINF)
                
                if not self._endswith(ids, [self.newline]):
                    scores[0, self.newline] = 0.0
                else:
                    scores[0, self.think_end[0]] = 0.0
                    self.in_think = False
        
        return scores


# ──────────────────────────────────────────────────────────────────────────
# 3.  Utility: split reasoning / answer (optional)
# ──────────────────────────────────────────────────────────────────────────
_tag_re = re.compile(
    r"<think>(.*?)</think>.*?<answer>(.*?)</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)


def split_reasoning_answer(text: str, tokenizer=None):
    """Split the generated text into reasoning and answer parts."""

    
    reasoning = None
    answer = None
    
    # Since <think> is in the prompt, we need to extract everything before </think>
    if "</think>" in text:
        think_end = text.find("</think>")
        reasoning = text[:think_end].strip()
        
        # The rest of the text after </think>
        remaining_text = text[think_end + 8:].strip()  # +8 for len("</think>")
        
        # Extract answer from remaining text
        if "<answer>" in remaining_text:
            answer_start = remaining_text.find("<answer>")
            answer_end = remaining_text.find("</answer>")
            if answer_end > answer_start:
                answer = remaining_text[answer_start + 8:answer_end].strip()  # +8 for len("<answer>")
            else:
                answer = remaining_text[answer_start + 8:].strip()
        else:
            # If no <answer> tag, treat remaining text as answer
            answer = remaining_text.strip()
    else:
        # Fallback: if no </think> found, check for <answer> tags
        if "<answer>" in text:
            answer_start = text.find("<answer>")
            answer_end = text.find("</answer>")
            if answer_end > answer_start:
                answer = text[answer_start + 8:answer_end].strip()
            else:
                answer = text[answer_start + 8:].strip()
        else:
            # Last resort: treat entire text as answer
            answer = text.strip()
    
    reasoning_tokens = len(tokenizer.encode(reasoning)) if reasoning and tokenizer else 0
    answer_tokens = len(tokenizer.encode(answer)) if answer and tokenizer else 0

    
    return reasoning, answer, reasoning_tokens, answer_tokens

# ──────────────────────────────────────────────────────────────────────────
# 4.  Data loading
# ──────────────────────────────────────────────────────────────────────────
def load_dataset(path: str = "prompts.json", limit: int = -1, seed: int = 42):
    data = json.loads(Path(path).read_text(encoding="utf‑8"))
    prompts = [d["prompt"] for d in data]
    conflicts = [d.get("conflict") for d in data]
    rng = random.Random(seed)
    shuff = list(zip(prompts, conflicts))
    rng.shuffle(shuff)
    if limit > 0:
        shuff = shuff[:limit]
    return [p for p, _ in shuff], [c for _, c in shuff]


# ──────────────────────────────────────────────────────────────────────────
# 5.  Worker: generate one response
# ──────────────────────────────────────────────────────────────────────────
async def generate_one(
    model,
    tokenizer,
    question: str,
    conflict: Any,
    think_budget: int,
    n: int,
    temperature: float,
    top_p: float,
    max_new: int,
    prompt_template: PromptTemplate,
    executor,
    timeout: int = 300,
):
    prompt = prompt_template.generate(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # A fresh processor per call
    processor = ThinkingTokenBudgetProcessor(tokenizer, think_budget)
    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    loop = asyncio.get_running_loop()

    try:
        outputs = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                lambda: model.generate(
                    **inputs, 
                    generation_config=gen_cfg, 
                    logits_processor=[processor],
                    use_cache=True,
                ),
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return dict(question=question, response=None, reasoning=None, conflict=conflict, 
                   reasoning_tokens=0, response_tokens=0)
    except Exception as e:
        return dict(question=question, response=None, reasoning=None, conflict=conflict,
                   reasoning_tokens=0, response_tokens=0)

    # FIXED: Decode the full output, then extract what comes after "Assistant:"
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response part
    if "Assistant:" in full_text:
        new_text = full_text.split("Assistant:", 1)[1].strip()
    else:
        # Fallback: decode only new tokens as before
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        new_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    
    reasoning, answer, reasoning_tokens, response_tokens = split_reasoning_answer(new_text, tokenizer)
    return dict(question=question, response=answer, reasoning=reasoning, conflict=conflict,
               reasoning_tokens=reasoning_tokens, response_tokens=response_tokens)


# ──────────────────────────────────────────────────────────────────────────
# 6.  Main async loop
# ──────────────────────────────────────────────────────────────────────────
async def run(
    model_name: str,
    prompts_path: str,
    batch: int,
    think_tokens: int,
    n_gen: int,
    temperature: float,
    top_p: float,
    max_new: int,
    output_dir: Path,
    row_limit: int
):
    print(f"Loading model {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    prompts, conflicts = load_dataset(prompts_path, limit=row_limit)
    print(f"Loaded {len(prompts)} prompts")
    
    template = PromptTemplate()

    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{model_name.replace('/', '_')}_think{think_tokens}.jsonl"

    # Use a single ThreadPoolExecutor for CPU‑side tasks
    from concurrent.futures import ThreadPoolExecutor

    # Track statistics
    all_reasoning_tokens = []
    all_response_tokens = []

    with ThreadPoolExecutor(max_workers=1) as pool:
        # Create progress bar for batches
        batch_count = (len(prompts) + batch - 1) // batch
        batch_pbar = tqdm(range(0, len(prompts), batch), desc="Processing batches", unit="batch")
        
        for i in batch_pbar:
            batch_prompts = prompts[i : i + batch]
            batch_conflicts = conflicts[i : i + batch]

            tasks = [
                generate_one(
                    model,
                    tokenizer,
                    q,
                    c,
                    think_tokens,
                    n_gen,
                    temperature,
                    top_p,
                    max_new,
                    template,
                    pool,
                )
                for q, c in zip(batch_prompts, batch_conflicts)
            ]
            results = await asyncio.gather(*tasks)

            # Collect token statistics for this batch
            batch_reasoning_tokens = []
            batch_response_tokens = []
            
            with outfile.open("a", encoding="utf‑8") as f:
                for obj in results:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    if obj.get('reasoning_tokens', 0) > 0:
                        batch_reasoning_tokens.append(obj['reasoning_tokens'])
                        all_reasoning_tokens.append(obj['reasoning_tokens'])
                    if obj.get('response_tokens', 0) > 0:
                        batch_response_tokens.append(obj['response_tokens'])
                        all_response_tokens.append(obj['response_tokens'])

            # Calculate and display batch statistics
            avg_reasoning = sum(batch_reasoning_tokens) / len(batch_reasoning_tokens) if batch_reasoning_tokens else 0
            avg_response = sum(batch_response_tokens) / len(batch_response_tokens) if batch_response_tokens else 0
            
            batch_pbar.set_postfix({
                'reasoning_tokens': f'{avg_reasoning:.1f}',
                'response_tokens': f'{avg_response:.1f}',
                'processed': f'{i+len(batch_prompts)}/{len(prompts)}'
            })

    # Final statistics
    if all_reasoning_tokens and all_response_tokens:
        print(f"\nFinal Statistics:")
        print(f"Average reasoning tokens: {sum(all_reasoning_tokens) / len(all_reasoning_tokens):.1f}")
        print(f"Average response tokens: {sum(all_response_tokens) / len(all_response_tokens):.1f}")
        print(f"Total samples processed: {len(all_reasoning_tokens)}")
    
    print(f"Results saved to: {outfile}")


# ──────────────────────────────────────────────────────────────────────────
# 7.  CLI entry‑point
# ──────────────────────────────────────────────────────────────────────────
def cli():
    p = argparse.ArgumentParser(description="HF chat with thinking‑token budget")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    p.add_argument("--prompts", default="prompts.json")
    p.add_argument("--output", default="safetyresults")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--think-tokens", type=int, default=100)
    p.add_argument("--n", type=int, default=1, help="num return sequences")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new", type=int, default=1500)
    p.add_argument("--limit", type=int, default=-1,
                   help="use at most this many rows from the dataset (-1 = all)")
    args = p.parse_args()

    try:
        asyncio.run(
            run(
                model_name=args.model,
                prompts_path=args.prompts,
                batch=args.batch,
                think_tokens=args.think_tokens,
                n_gen=args.n,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new=args.max_new,
                output_dir=Path(args.output),
                row_limit=args.limit
            )
        )
    except KeyboardInterrupt:
        print("Interrupted — partial results saved.")

if __name__ == "__main__":
    cli()