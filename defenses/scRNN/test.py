from corrector import ScRNNChecker
checker = ScRNNChecker()
a= checker.correct_string("nicset atcing I have ever witsesed")
print(a)


import datetime

import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertConfig, DataCollatorWithPadding, get_scheduler
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DistilBertForSequenceClassification
from noised_words import noise_string, rnd_noise_sentence, rnd_noise_sentence_word
import sys

RUN_LOG = f"RunLog-{datetime.datetime.now().date()}.txt"

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]


def train_and_eval(task_name='cola', epochs=5, noise_train=0.0, noises_val=None, log_file=None):
    task = task_name
    sentence1_key, sentence2_key = task_to_keys[task]

    if noises_val is None:
        noises_val = [0.0]

    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    def tokenize_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key],padding=True, truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key],padding=True, truncation=True,)

    def preprocess_noise_function_train(examples):
        if sentence2_key is None:
            return rnd_noise_sentence_word(examples[sentence1_key], None, sentence1_key, sentence2_key,
                                           rate=noise_train)
        return rnd_noise_sentence_word(examples[sentence1_key], examples[sentence2_key], sentence1_key, sentence2_key,
                                       rate=noise_train)


    batch_size = 16
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    dataset = dataset.rename_column("label", "labels")


    noised_dataset = dataset['train'].map(preprocess_noise_function_train, batched=False)
    tokenized_train = noised_dataset.map(tokenize_function, batched=False)
    if sentence2_key is None:
        tokenized_train = tokenized_train.remove_columns([sentence1_key, "idx"])
    else:
        tokenized_train = tokenized_train.remove_columns([sentence1_key, sentence2_key, "idx"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=16, collate_fn=data_collator)

    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    dataset_eval = dataset[validation_key]

    metric = load_metric('glue', actual_task)

    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

    # Pretrained bert
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

    model_name = model_checkpoint.split("/")[-1]

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=2e-5)
    num_training_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        losses = np.empty(0)
        for index, batch in enumerate(train_dataloader):
            if index % 400 == 0:
                print(f'progress is: {index / len(train_dataloader)}', file=sys.stderr)

            for key in [sentence1_key, sentence2_key, 'idx']:
                batch.pop(key, None)
            batch['input_ids'] = torch.tensor(batch['input_ids'])
            batch['attention_mask'] = torch.tensor(batch['attention_mask'])
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            losses = np.append(losses, loss.item())
            loss.backward()
            for _, v in batch.items():
                del v
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch} loss is {np.mean(losses)}')

    val_dataloader = DataLoader(dataset[validation_key], shuffle=False, batch_size=16)
    for noise_val in noises_val:
        for batch in val_dataloader:
            for i in range(len(batch)):
                if sentence2_key is not None:
                    noise_dict = rnd_noise_sentence_word(batch[sentence1_key][i], batch[sentence2_key][i],
                                                         sentence1_key,
                                                         sentence2_key,
                                                         rate=noise_val)
                else:
                    noise_dict = rnd_noise_sentence_word(batch[sentence1_key][i], None,
                                                         sentence1_key,
                                                         sentence2_key,
                                                         rate=noise_val)
                batch[sentence1_key][i] = noise_dict[sentence1_key]
                if sentence2_key is not None:
                    batch[sentence2_key][i] = noise_dict[sentence2_key]
            if sentence2_key is not None:
                toks = tokenizer(batch[sentence1_key], batch[sentence2_key], truncation=True, padding=True)
            else:
                toks = tokenizer(batch[sentence1_key], truncation=True, padding=True)
            batch.update(toks)
            for key in [sentence1_key, sentence2_key, 'idx']:
                batch.pop(key, None)
            batch['input_ids'] = torch.tensor(batch['input_ids'])
            batch['attention_mask'] = torch.tensor(batch['attention_mask'])
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            for _, v in batch.items():
                del v

        met = metric.compute()
        print(f'noise train: {noise_train} and noise eval: {noise_val}')
        print(f'task {task} metric after {epochs} epochs is :')
        print(met)
    # torch.save(model.state_dict(), f'task_{task}_noise_{noise_train}_epochs_{epochs}.pt')
    # print_log(f'Task: {task_name} with noise: {noise} Metric:', log_file)
    # print_log(metric, log_file)


def print_log(log_message, log_file):
    print(log_message)
    log_file.write(str(log_message))
    log_file.write("\n")


if __name__ == '__main__':
    noises_train = [0.0, 0.2, 0.4, 0.6, 0.8, 1]
    noises_val = [0.0, 0.2, 0.4, 0.6, 0.8, 1]
    tasks = ['mrpc','sst2', 'qqp']
    for task in tasks:
        for noise in noises_train:
            print(f'Task: {task} with noise: {noise} Training')
            train_and_eval(task, 5, noise, noises_val)
