import argparse
import json
import random
from pathlib import Path
from typing import Any, List
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessor,
)
from tqdm import tqdm

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
        "Assistant: <think>\n"
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
# 2.  Batched Thinking-token budget processor
# ──────────────────────────────────────────────────────────────────────────
class BatchedThinkingTokenBudgetProcessor(LogitsProcessor):
    """
    Batched version that handles multiple sequences simultaneously.
    Each sequence gets its own budget tracking.
    """

    def __init__(self, tokenizer, max_thinking_tokens: int, batch_size: int):
        super().__init__()
        self.tok = tokenizer
        self.max_tokens = max_thinking_tokens
        self.batch_size = batch_size

        # Tag tokens
        self.think_end = tokenizer.encode("</think>", add_special_tokens=False)  
        self.newline = tokenizer.encode("\n", add_special_tokens=False)[0]

        # Runtime state per sequence - start all in think mode
        self.in_think = [True] * batch_size
        self.generation_steps = [0] * batch_size
        self.forcing_closure = [False] * batch_size
        self.already_forced = [False] * batch_size  # Track if we already forced closure
        self.call_count = 0  # Track total processor calls
        self.NINF = float("-inf")
        
        print(f"DEBUG: Batched processor initialized for {batch_size} sequences, max tokens: {max_thinking_tokens}")

    def _endswith(self, ids: List[int], suffix: List[int]) -> bool:
        return len(ids) >= len(suffix) and ids[-len(suffix) :] == suffix

class BatchedThinkingTokenBudgetProcessor(LogitsProcessor):
    """
    Batched version that handles multiple sequences simultaneously.
    Each sequence gets its own budget tracking.
    """

    def __init__(self, tokenizer, max_thinking_tokens: int, batch_size: int):
        super().__init__()
        self.tok = tokenizer
        self.max_tokens = max_thinking_tokens
        self.batch_size = batch_size

        # Tag tokens
        self.think_end = tokenizer.encode("</think>", add_special_tokens=False)  
        self.newline = tokenizer.encode("\n", add_special_tokens=False)[0]

        # Runtime state per sequence - start all in think mode
        self.in_think = [True] * batch_size
        self.generation_steps = [0] * batch_size
        self.forcing_closure = [False] * batch_size
        self.already_forced = [False] * batch_size  # Track if we already forced closure
        self.call_count = 0  # Track total processor calls
        self.NINF = float("-inf")
        
        print(f"DEBUG: Batched processor initialized for {batch_size} sequences, max tokens: {max_thinking_tokens}")

    def _endswith(self, ids: List[int], suffix: List[int]) -> bool:
        return len(ids) >= len(suffix) and ids[-len(suffix) :] == suffix

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Increment call count once at the start
        self.call_count += 1
        actual_batch_size = input_ids.shape[0]
        
        # Early exit if all sequences finished thinking - FAST PATH
        active_thinking = sum(1 for x in self.in_think if x)
        if active_thinking == 0:
            if self.call_count % 100 == 1:  # Occasional debug
                print(f"DEBUG: All sequences finished thinking, processor idle (call #{self.call_count})")
            return scores  # Return immediately - no expensive work!
        
        # EXPENSIVE WORK ONLY RUNS WHEN SEQUENCES ARE STILL THINKING
        
        # Debug info every 10 calls to verify batching
        if self.call_count % 10 == 1:
            print(f"DEBUG: Processor call #{self.call_count} - Processing {actual_batch_size} sequences simultaneously")
            if actual_batch_size != self.batch_size:
                print(f"WARNING: Expected batch size {self.batch_size}, got {actual_batch_size}")
        
        # Process each sequence in the batch
        for batch_idx in range(actual_batch_size):
            if batch_idx >= len(self.in_think):
                continue
            
            # Use generation steps instead of prompt lengths to avoid padding issues
            self.generation_steps[batch_idx] += 1
            ids = input_ids[batch_idx].tolist()
            
            if self.in_think[batch_idx]:
                # Check if we naturally hit </think>
                if self._endswith(ids, self.think_end):
                    if not self.already_forced[batch_idx]:
                        print(f"DEBUG: Seq {batch_idx} - Natural </think> at step {self.generation_steps[batch_idx]} (call #{self.call_count})")
                    else:
                        print(f"DEBUG: Seq {batch_idx} - Forced </think> executed at step {self.generation_steps[batch_idx]} (call #{self.call_count})")
                    self.in_think[batch_idx] = False
                    self.forcing_closure[batch_idx] = False
                    continue
                
                # Check if we've hit our budget
                if self.generation_steps[batch_idx] >= self.max_tokens and not self.forcing_closure[batch_idx]:
                    print(f"DEBUG: Seq {batch_idx} - Hit budget at step {self.generation_steps[batch_idx]}, forcing </think> (call #{self.call_count})")
                    self.forcing_closure[batch_idx] = True
                    self.already_forced[batch_idx] = True
                
                # If we're forcing closure, manipulate the logits for this sequence
                if self.forcing_closure[batch_idx]:
                    scores[batch_idx].fill_(self.NINF)
                    
                    if not self._endswith(ids, [self.newline]):
                        scores[batch_idx, self.newline] = 0.0
                    else:
                        scores[batch_idx, self.think_end[0]] = 0.0
        
        # Summary every 50 calls
        if self.call_count % 50 == 0:
            finished_sequences = sum(1 for x in self.in_think[:actual_batch_size] if not x)
            print(f"DEBUG: Call #{self.call_count} Summary - {finished_sequences}/{actual_batch_size} sequences finished thinking")

        return scores

def split_reasoning_answer(text: str, tokenizer=None):
    """Split the generated text into reasoning and answer parts."""
    reasoning = None
    answer = None
    
    # Now that <think> is in the prompt, the generated text should start with reasoning content
    if "</think>" in text:
        think_end = text.find("</think>")
        # If text starts with <think>, remove it; otherwise extract everything before </think>
        if text.startswith("<think>"):
            reasoning = text[7:think_end].strip()  # Remove <think> and extract content
        else:
            reasoning = text[:think_end].strip()
        
        # The rest after </think>
        remaining_text = text[think_end + 8:].strip()
        
        # Extract answer
        if "<answer>" in remaining_text:
            answer_start = remaining_text.find("<answer>")
            answer_end = remaining_text.find("</answer>")
            if answer_end > answer_start:
                answer = remaining_text[answer_start + 8:answer_end].strip()
            else:
                answer = remaining_text[answer_start + 8:].strip()
        else:
            answer = remaining_text.strip()
    else:
        # Fallback if no </think> found
        if "<answer>" in text:
            answer_start = text.find("<answer>")
            answer_end = text.find("</answer>")
            if answer_end > answer_start:
                answer = text[answer_start + 8:answer_end].strip()
            else:
                answer = text[answer_start + 8:].strip()
        else:
            answer = text.strip()
    
    reasoning_tokens = len(tokenizer.encode(reasoning)) if reasoning and tokenizer else 0
    answer_tokens = len(tokenizer.encode(answer)) if answer and tokenizer else 0
    
    return reasoning, answer, reasoning_tokens, answer_tokens

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

def generate_batch_sync(
    model,
    tokenizer,
    questions: List[str],
    conflicts: List[Any],
    think_budget: int,
    temperature: float,
    top_p: float,
    max_new: int,
    prompt_template: PromptTemplate,
):
    # Generate prompts for all questions
    prompts = [prompt_template.generate(q) for q in questions]
    
    # Tokenize with padding to longest sequence
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        add_special_tokens=False,
    ).to(model.device)  # Use model.device
    
    # Create thinking budget processor
    batch_size = len(questions)
    processor = BatchedThinkingTokenBudgetProcessor(tokenizer, think_budget, batch_size)
    
    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # DIRECT SYNCHRONOUS GENERATION
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_cfg,
            logits_processor=[processor],  # Add back the thinking budget processor
            use_cache=True,
        )

    # Process results
    results = []
    for i, (question, conflict) in enumerate(zip(questions, conflicts)):
        full_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        
        if "Assistant: <think>" in full_text:
            new_text = full_text.split("Assistant: <think>", 1)[1]
            new_text = "<think>" + new_text
        else:
            new_text = full_text
        
        reasoning, answer, reasoning_tokens, response_tokens = split_reasoning_answer(new_text, tokenizer)
        results.append(dict(
            question=question,
            response=answer,
            reasoning=reasoning,
            conflict=conflict,
            reasoning_tokens=reasoning_tokens,
            response_tokens=response_tokens
        ))
    
    return results

def run_sync(
    model_name: str,
    prompts_path: str,
    batch: int,
    think_tokens: int,
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
    
    # Create timestamped output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split('/')[-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{model_short}_think{think_tokens}_{timestamp}.jsonl"
    
    print(f"Output file: {outfile}")
    if outfile.exists():
        outfile.unlink()

    # Process batches
    all_reasoning_tokens = []
    all_response_tokens = []
    
    for i in tqdm(range(0, len(prompts), batch), desc="Processing batches"):
        batch_prompts = prompts[i : i + batch]
        batch_conflicts = conflicts[i : i + batch]

        # Direct synchronous call
        results = generate_batch_sync(
            model,
            tokenizer,
            batch_prompts,
            batch_conflicts,
            think_tokens,
            temperature,
            top_p,
            max_new,
            template,
        )

        # Save results
        with outfile.open("a", encoding="utf‑8") as f:
            for obj in results:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                if obj.get('reasoning_tokens', 0) > 0:
                    all_reasoning_tokens.append(obj['reasoning_tokens'])
                if obj.get('response_tokens', 0) > 0:
                    all_response_tokens.append(obj['response_tokens'])

    # Final statistics
    if all_reasoning_tokens and all_response_tokens:
        print(f"\nFinal Statistics:")
        print(f"Average reasoning tokens: {sum(all_reasoning_tokens) / len(all_reasoning_tokens):.1f}")
        print(f"Average response tokens: {sum(all_response_tokens) / len(all_response_tokens):.1f}")
    
    print(f"Results saved to: {outfile}")

def cli():
    p = argparse.ArgumentParser(description="HF chat with GPU batching and thinking-token budget")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    p.add_argument("--prompts", default="prompts.json")
    p.add_argument("--output", default="safetyresults")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--think-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new", type=int, default=1500)
    p.add_argument("--limit", type=int, default=-1,
                   help="use at most this many rows from the dataset (-1 = all)")
    args = p.parse_args()

    try:
        run_sync(
            model_name=args.model,
            prompts_path=args.prompts,
            batch=args.batch,
            think_tokens=args.think_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new=args.max_new,
            output_dir=Path(args.output),
            row_limit=args.limit
        )
    except KeyboardInterrupt:
        print("Interrupted — partial results saved.")

if __name__ == "__main__":
    cli()