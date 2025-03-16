"""WAENING: NOT TESTED YET!"""

import json
import difflib
import subprocess
import os
import tempfile
from tqdm import tqdm

OUTPUT_PATH = "reasoning_traces.jsonl"
EXPECTED_OUTPUTS_PATH = "expected_outputs.jsonl"
EVALUATION_RESULTS_PATH = "evaluation_results.jsonl"
SUCCESS_CASES_PATH = "success_cases.jsonl"
FAILURE_CASES_PATH = "failure_cases.jsonl"

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def extract_code(text):
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.strip().startswith('```python'):
            in_code = True
            continue
        elif line.strip() == '```' and in_code:
            in_code = False
            continue
        elif in_code:
            code_lines.append(line)
    
    if not code_lines:
        return text
    
    return '\n'.join(code_lines)

def evaluate_results(generated, expected):
    results = []
    success_cases = []
    failure_cases = []
    
    for gen, exp in tqdm(zip(generated, expected), total=len(generated)):
        prompt = gen["prompt"]
        generated_output = gen["full_response"]
        gold_standard = exp["gold_standard_solution"]
        
        extracted_code = extract_code(generated_output)
        
        solution_similarity = difflib.SequenceMatcher(None, extracted_code, gold_standard).ratio()
        solution_match = solution_similarity > 0.8
        
        test_results = []
        all_tests_pass = True
        
        if "verification_info" in exp and "test_cases" in exp["verification_info"]:
            for test_case in exp["verification_info"]["test_cases"]:
                if test_case.get("type") == "stdin_stdout":
                    expected_output = test_case["output"].strip()
                    test_input = test_case.get("input", "")
                    
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
                            temp_file.write(extracted_code.encode())
                            temp_file_path = temp_file.name
                        
                        process = subprocess.run(
                            ["python", temp_file_path],
                            input=test_input,
                            text=True,
                            capture_output=True,
                            timeout=10
                        )
                        
                        actual_output = process.stdout.strip()
                        test_passed = actual_output == expected_output
                        
                        test_results.append({
                            "expected": expected_output,
                            "actual": actual_output,
                            "passed": test_passed
                        })
                        
                        if not test_passed:
                            all_tests_pass = False
                            
                    except Exception as e:
                        test_results.append({
                            "expected": expected_output,
                            "actual": f"Error: {str(e)}",
                            "passed": False
                        })
                        all_tests_pass = False
                    finally:
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
        
        result_entry = {
            "prompt": prompt,
            "solution_similarity": solution_similarity,
            "solution_match": solution_match,
            "generated_code": extracted_code,
            "gold_standard": gold_standard,
            "test_results": test_results,
            "all_tests_pass": all_tests_pass
        }
        
        results.append(result_entry)
        
        if solution_match or all_tests_pass:
            success_cases.append(result_entry)
        else:
            failure_cases.append(result_entry)
    
    return results, success_cases, failure_cases

def save_results(results, file_path):
    with open(file_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

generated_outputs = load_jsonl(OUTPUT_PATH)
expected_outputs = load_jsonl(EXPECTED_OUTPUTS_PATH)

evaluation_results, success_cases, failure_cases = evaluate_results(generated_outputs, expected_outputs)

save_results(evaluation_results, EVALUATION_RESULTS_PATH)
save_results(success_cases, SUCCESS_CASES_PATH)
save_results(failure_cases, FAILURE_CASES_PATH)

print(f"Evaluation completed. Results saved to {EVALUATION_RESULTS_PATH}")
print(f"Success cases: {len(success_cases)}/{len(evaluation_results)} ({len(success_cases)/len(evaluation_results)*100:.2f}%)")
print(f"Failure cases: {len(failure_cases)}/{len(evaluation_results)} ({len(failure_cases)/len(evaluation_results)*100:.2f}%)")
