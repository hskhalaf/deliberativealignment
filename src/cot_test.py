import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel
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

data_dict = {"text": reasonings, "label": new_labels}

raw_dataset = Dataset.from_dict(data_dict)

features = raw_dataset.features.copy()
features["label"] = ClassLabel(names=["negative", "positive"])  # 혹은 0/1이면 이렇게

raw_dataset = raw_dataset.cast(features)
raw_dataset = raw_dataset.rename_column("label", "labels")


# split
train_valtest = raw_dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="labels")
val_test = train_valtest['test'].train_test_split(test_size=0.5, seed=42, stratify_by_column="labels")

# wrap in DatasetDict
dataset = DatasetDict({
    'train': train_valtest['train'],
    'validation': val_test['train'],
    'test': val_test['test'],
})

# ========== 2. Loading Llama and tokenizer ==========

# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # LLaMA doesn't have pad, 

base_model = LlamaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    torch_dtype=torch.float16,
    device_map="auto"  
)
base_model.config.pad_token_id = tokenizer.pad_token_id

# ========== 3. Setting Lora ==========
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Set proper modules for Llama
)

model = get_peft_model(base_model, peft_config)
model = model.bfloat16()
# ========== 4. Preprocessing dataset ==========
# max_length = 2048
max_length = 8192

dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length), batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_dataset = dataset["train"].shuffle(seed=42)  # for fast example
eval_dataset = dataset["test"] #.select(range(500))


# ========== 5. TrainingArguments 설정 ==========
training_args = TrainingArguments(
    output_dir="./llama-binary-cls",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    max_grad_norm=1.0,
    logging_steps=10,
    save_total_limit=2,
    fp16=False,
    bf16=True,
    report_to="none"
)

from collections import Counter
# print(Counter([len(t.split()) for t in train_dataset['text']]))  # 길이 분포 확인

        
# sample = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=1)))
# print(sample["input_ids"].dtype)  # => torch.int64
# print(sample["attention_mask"].dtype)
# print(sample["labels"])           # => Should be int64 or long
# ========== 6. Trainer 정의 및 학습 ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
