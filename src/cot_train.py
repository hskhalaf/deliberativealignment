import pandas as pd
from datasets import Dataset
from transformers import LlamaTokenizer, LlamaForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 1. Loading a datset ==========

path = 'results/deepseek-ai/'
model = 'deepseek-r1-distill-qwen-14b'
hash_map = {}
cnt = 0
reasonings = []

with open(path + model + '_label.pkl', 'rb') as f:
    labels = pickle.load(f)

new_labels = []
with open(path + model + '.jsonl', 'r') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        question, answer, output, reasoning = data['question'], data['expected_output'], data['response'], data['reasoning']
        if question == None or output == None or reasoning == None:
            continue
        cnt += 1
        reasonings.append(data['reasoning'])
        new_labels.append(labels[cnt])
        
# print(labels)
# print(len(reasonings))
# print(len(labels.keys()))

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token  # 필요 시

token_cnts = []
for reasoning in reasonings:
    token_cnts.append(len(tokenizer.tokenize(reasoning)))
    
print(token_cnts)
# df["token_count"] = df["text"].apply(lambda x: len(tokenizer.tokenize(x)))

# 2. 특정 단어 등장 여부 및 횟수
target_word = "wait"
wait_cnts = []
for reasoning in reasonings:
    token_cnts.append(len(tokenizer.tokenize(reasoning)))
print(wait_cnts)
# df["wait_count"] = df["text"].str.lower().str.count(rf"\b{target_word}\b")
# df["has_wait"] = df["wait_count"] > 0

# cnt = 0 
# lenghts = []
# for i, reasoning in enumerate(reasonings):
#     if reasoning != None and 'wait' in reasoning:
#         # print(reasoning)
#         cnt += 1
#     if reasoning != None:
#         lenghts.append(len(reasoning))

# print('# of data :', len(outputs))
# output_codes = []
# for i, (question, answer, output) in enumerate(zip(questions, answers, outputs)):
#     # print(question)
#     # print(answer)
#     output_code = extract_code_blocks(output)
#     output_codes.append(output_code)
#     # print(output_code)
#     # print(answer['test_cases'][0])
# results, metadata = evaluate_generations(answers,output_codes, debug=False)
# cnt = 0
# print(results)
# labels = {}
# for i in range(len(results)):
#     labels[i] = 0
# for key, value in results.items():
#     if value[0][0]  == True:
#         cnt += 1
#         labels[hash_map[key]] = 1
# print(f'{cnt}/{len(results)}')

# with open(path + model + '_label.pkl', "wb") as f:
#     pickle.dump(labels, f)
# def load_data(path):
#     df = pd.read_csv(path)
#     return Dataset.from_pandas(df)

# train_dataset = load_data("train.csv")
# eval_dataset = load_data("test.csv")

# # ========== 2. 모델/토크나이저 불러오기 ==========
# model_name = "meta-llama/Llama-2-7b-hf"  # HuggingFace 허용된 모델이어야 함
# tokenizer = LlamaTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token  # LLaMA는 pad_token 없음, eos로 대체

# model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# # ========== 3. LoRA 설정 ==========
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS, 
#     inference_mode=False,
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.1,
#     target_modules=["q_proj", "v_proj"]  # LLaMA에 맞는 모듈 지정
# )

# model = get_peft_model(model, peft_config)

# # ========== 4. 데이터 전처리 ==========
# max_length = 512

# def preprocess(example):
#     return tokenizer(
#         example["text"],
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#     )

# train_dataset = train_dataset.map(preprocess, batched=True)
# eval_dataset = eval_dataset.map(preprocess, batched=True)

# # column 정리
# columns = ["input_ids", "attention_mask", "label"]
# train_dataset.set_format("torch", columns=columns)
# eval_dataset.set_format("torch", columns=columns)

# # ========== 5. TrainingArguments 설정 ==========
# training_args = TrainingArguments(
#     output_dir="./llama-binary-cls",
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=3,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=2e-4,
#     logging_steps=10,
#     save_total_limit=2,
#     fp16=True,
#     report_to="none"
# )

# # ========== 6. Trainer 정의 및 학습 ==========
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
# )

# trainer.train()