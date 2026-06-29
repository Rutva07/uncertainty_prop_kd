import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description="Teacher Output Uncertainty Script")
parser.add_argument("--teacher_temperature", type=float, default=1.0, help="Temperature used in teacher generation")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for generation")
#parser.add_argument("--teacher_path", type=str, default="", help="Teacher model path")
#parser.add_argument("--student_out_dir", type=str, default="", help="Student output path")
parser.add_argument("--teacher_path", type=str, default="gpt2", help="Teacher model path")
parser.add_argument("--student_out_dir", type=str, default="outputs/students", help="Student output path")
args = parser.parse_args()

teacher_temperature = args.teacher_temperature
batch_size = args.batch_size
teacher_model_path = args.teacher_path
output_dir = args.student_out_dir

# Settings
device = "cuda" if torch.cuda.is_available() else "cpu"
n = 10  # number of student models
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
max_len = tokenizer.model_max_length
training_data_path = "data/training_data.json"


# Load teacher model and tokenizer
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
teacher_tokenizer.padding_side = "left"
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
teacher_model.to(device)
teacher_model.eval()

# Load sentence transformer
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")


# Load training data
with open(training_data_path) as f:
    raw_data = json.load(f)

def batch_generate_teacher_answers(prompts, temperature, batch_size, top_k=50):
    all_answers = []
    all_embeddings = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating teacher answers"):
        batch_prompts = prompts[i:i + batch_size]
        input_texts = [f"Question: {p}\nAnswer:" for p in batch_prompts]

        encodings = teacher_tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(device)

        with torch.no_grad():
            outputs = teacher_model.generate(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
                max_length=max_len,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=1.2,
                pad_token_id=teacher_tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for input_ids, output_ids in zip(encodings.input_ids, outputs):
            answer_ids = output_ids[len(input_ids):]
            answer = teacher_tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            embedding = sent_transformer.encode([answer])[0]
            all_answers.append(answer)
            all_embeddings.append(embedding.tolist())

    return all_answers, all_embeddings

def preprocess(example):
    q_text = f"Question: {example['question']}\n"
    a_text = f"Answer: {example['answer']}"
    
    q_tokens = tokenizer(q_text, add_special_tokens=False)
    a_tokens = tokenizer(a_text, add_special_tokens=False)

    input_ids = q_tokens["input_ids"] + a_tokens["input_ids"]
    input_ids = input_ids[:max_len]  # truncate if needed

    attention_mask = [1] * len(input_ids)

    labels = [-100] * len(q_tokens["input_ids"]) + a_tokens["input_ids"]
    labels = labels[:max_len]

    pad_len = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len
    labels += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# Distill n student models
embedding_record = {}

for student_idx in range(n):
    print(f"Distilling student {student_idx+1}/{n}")

    question_list = raw_data["questions"]

    questions = [item["body"] for item in question_list]
    answers, embeddings = batch_generate_teacher_answers(
        questions,
        teacher_temperature,
        batch_size=batch_size
    )

    distilled_data = []
    for i, item in enumerate(question_list):
        distilled_data.append({
            "question": item["body"],
            "answer": answers[i]
        })

        if item["body"] not in embedding_record:
            embedding_record[item["body"]] = []

        embedding_record[item["body"]].append(embeddings[i])

    # Build dataset
    dataset = Dataset.from_list(distilled_data)
    
    # Use the same tokenizer for teacher and student
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = dataset.map(preprocess, remove_columns=["question", "answer"])
    
    # Load student model with fixed seed
    torch.manual_seed(1)
    random.seed(1)
    student = AutoModelForCausalLM.from_pretrained("distilgpt2")

    # Setup training
    student_out_dir = os.path.join(output_dir, f"student_{student_idx}")
    training_args = TrainingArguments(
        output_dir=student_out_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=10,
        logging_steps=1000,
        save_total_limit=1,
        learning_rate=5e-5,
        report_to="none"
    )

    trainer = Trainer(
        model=student,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    )

    # Train and save
    trainer.train()
    trainer.save_model(student_out_dir)

    print(f"Saved student {student_idx+1} to {student_out_dir}")