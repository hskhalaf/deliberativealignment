import argparse
import json
import random
import math
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
# 2.  FIXED Thinking Token Budget Processor
# ──────────────────────────────────────────────────────────────────────────
class ThinkingTokenBudgetProcessor(LogitsProcessor):
    """
    LogitsProcessor that enforces a token budget for thinking sections.
    """
    
    def __init__(
        self, 
        max_thinking_tokens: int,
        think_end_token_ids: List[int],
        newline_token_id: int,
        initial_input_length: int,
        batch_size: int,
        device: torch.device
    ):
        self.max_thinking_tokens = max_thinking_tokens
        self.think_end_token_ids = torch.tensor(think_end_token_ids, device=device)
        self.newline_token_id = newline_token_id
        self.initial_input_length = initial_input_length
        
        # Track which sequences have finished thinking
        self.finished_thinking = [False] * batch_size
        self.call_count = 0
        
        print(f"DEBUG: Thinking processor initialized")
        print(f"DEBUG: Max tokens: {max_thinking_tokens}")
        print(f"DEBUG: Initial input length: {initial_input_length}")
        print(f"DEBUG: Think end tokens: {think_end_token_ids}")
        print(f"DEBUG: Batch size: {batch_size}")
    
    def _sequence_contains_think_end(self, sequence: torch.Tensor) -> bool:
        """Check if sequence contains </think> tokens"""
        think_end_len = len(self.think_end_token_ids)
        if len(sequence) < think_end_len:
            return False
            
        # Look for think_end pattern in the sequence
        sequence = sequence.to(self.think_end_token_ids.device)
        for i in range(think_end_len, len(sequence) + 1):
            if torch.equal(sequence[i-think_end_len:i], self.think_end_token_ids):
                return True
        return False
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.call_count += 1
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        
        # Calculate how many tokens have been generated
        tokens_generated = current_length - self.initial_input_length
        
        # Debug first few calls
        if self.call_count <= 3:
            print(f"DEBUG: Call #{self.call_count} - length: {current_length}, generated: {tokens_generated}")
        
        # Early exit if all sequences finished thinking
        if all(self.finished_thinking):
            if self.call_count % 100 == 1:
                print(f"DEBUG: All sequences finished thinking (call #{self.call_count})")
            return scores
        
        # Early exit if no tokens generated yet
        if tokens_generated <= 0:
            return scores
        
        # Process each sequence
        for i in range(batch_size):
            # Skip sequences that already finished thinking
            if self.finished_thinking[i]:
                continue
                
            sequence = input_ids[i]
            
            # Check if this sequence naturally ended thinking
            if self._sequence_contains_think_end(sequence):
                self.finished_thinking[i] = True
                print(f"DEBUG: Seq {i} - Natural </think> found at {tokens_generated} tokens - FINISHED")
                continue
            
            # Check if we need to force closure
            if tokens_generated >= self.max_thinking_tokens:
                print(f"DEBUG: Seq {i} - Hit budget ({tokens_generated}/{self.max_thinking_tokens}), forcing </think>")
                
                # Apply forced closure
                scores[i] = torch.full_like(scores[i], -math.inf)
                
                # Force newline first, then </think>
                if len(sequence) > 0 and sequence[-1] != self.newline_token_id:
                    scores[i, self.newline_token_id] = 0.0
                else:
                    scores[i, self.think_end_token_ids[0]] = 0.0
        
        return scores


def create_thinking_budget_processor(
    tokenizer, 
    max_thinking_tokens: int, 
    padded_inputs: torch.Tensor
) -> ThinkingTokenBudgetProcessor:
    """Create a thinking budget processor for the given batch."""
    
    # Get token IDs
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    newline_token = tokenizer.encode("\n", add_special_tokens=False)[0]
    
    initial_input_length = padded_inputs.shape[1]
    batch_size = padded_inputs.shape[0]
    device = padded_inputs.device
    
    return ThinkingTokenBudgetProcessor(
        max_thinking_tokens=max_thinking_tokens,
        think_end_token_ids=think_end_tokens,
        newline_token_id=newline_token,
        initial_input_length=initial_input_length,
        batch_size=batch_size,
        device=device
    )

# ──────────────────────────────────────────────────────────────────────────
# 3. FIXED Text processing functions
# ──────────────────────────────────────────────────────────────────────────
def split_reasoning_answer(text: str, tokenizer=None):
    """Split the generated text into reasoning and answer parts."""
    reasoning = None
    answer = None
    
    # Debug what we're parsing
    if len(text) > 50:
        print(f"DEBUG: Parsing text start: {repr(text[:100])}")
        print(f"DEBUG: Contains </think>: {'</think>' in text}")
        print(f"DEBUG: Contains <answer>: {'<answer>' in text}")
    
    # Look for </think> and <answer> tags
    if "</think>" in text and "<answer>" in text:
        # Split on </think>
        think_end = text.find("</think>")
        if text.startswith("<think>"):
            reasoning = text[7:think_end].strip()  # Remove <think> and extract content
        else:
            reasoning = text[:think_end].strip()
        
        # Extract answer part
        remaining_text = text[think_end + 8:].strip()
        answer_start = remaining_text.find("<answer>")
        if answer_start >= 0:
            answer_content = remaining_text[answer_start + 8:]
            answer_end = answer_content.find("</answer>")
            if answer_end >= 0:
                answer = answer_content[:answer_end].strip()
            else:
                answer = answer_content.strip()
        
    elif "</think>" in text:
        # Only reasoning, no answer section
        think_end = text.find("</think>")
        if text.startswith("<think>"):
            reasoning = text[7:think_end].strip()
        else:
            reasoning = text[:think_end].strip()
        answer = "No answer provided."
        
    elif "<answer>" in text:
        # Only answer, no reasoning
        answer_start = text.find("<answer>")
        answer_content = text[answer_start + 8:]
        answer_end = answer_content.find("</answer>")
        if answer_end >= 0:
            answer = answer_content[:answer_end].strip()
        else:
            answer = answer_content.strip()
        reasoning = "No reasoning provided."
        
    else:
        # No tags found - treat as raw response
        reasoning = "No reasoning provided."
        answer = text.strip()
    
    reasoning_tokens = len(tokenizer.encode(reasoning)) if reasoning and tokenizer else 0
    answer_tokens = len(tokenizer.encode(answer)) if answer and tokenizer else 0
    
    print(f"DEBUG: Parsed - reasoning: {len(reasoning or '')} chars, answer: {len(answer or '')} chars")
    
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

# ──────────────────────────────────────────────────────────────────────────
# 4. FIXED Generation functions
# ──────────────────────────────────────────────────────────────────────────
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
    ).to(model.device)
    
    # Create thinking budget processor with the actual padded inputs
    processor = create_thinking_budget_processor(
        tokenizer=tokenizer,
        max_thinking_tokens=think_budget,
        padded_inputs=inputs['input_ids']
    )
    
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
            logits_processor=[processor],
            use_cache=True,
        )

    # Process results
    results = []
    for i, (question, conflict) in enumerate(zip(questions, conflicts)):
        full_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        
        # FIXED: Extract generated content properly
        if "Assistant: <think>" in full_text:
            # Split and keep the generated part
            generated_part = full_text.split("Assistant: <think>", 1)[1]
            new_text = "<think>" + generated_part
        else:
            # Fallback
            new_text = full_text
        
        reasoning, answer, reasoning_tokens, response_tokens = split_reasoning_answer(new_text, tokenizer)
        results.append(dict(
            question=question,
            response=answer,  # This should be the clean answer now
            reasoning=reasoning,  # This should be the thinking content
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
        print(f"Total samples processed: {len(all_reasoning_tokens)}")
    
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