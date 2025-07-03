import logging
import warnings
import sys
from io import StringIO
from datasets import Dataset
from utils import load_rotten_tomatoes, get_dataset_splits
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, DataCollatorWithPadding,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

class EpochLogger(TrainerCallback):
 
    
    def __init__(self):
        self.epoch_count = 0
        self.header_printed = False
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
    
        if logs and 'eval_loss' in logs:
            if not self.header_printed:
                print(f"\n{'Epoch':<6} {'Validation Loss':<15} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1':<8}")
                print("-" * 65)
                self.header_printed = True
            
            self.epoch_count += 1
            print(f"{self.epoch_count:<6} {logs['eval_loss']:<15.6f} "
                  f"{logs['eval_accuracy']:<9.6f} {logs.get('eval_precision', 0.0):<10.6f} "
                  f"{logs.get('eval_recall', 0.0):<8.6f} {logs['eval_f1']:<8.6f}")

def prepare_datasets():
    
    dataset = load_rotten_tomatoes()
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_dataset_splits(dataset)
    
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    
    return train_dataset, val_dataset, test_dataset

def tokenize_datasets(train_dataset, val_dataset, test_dataset):
 
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)
    

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    return train_dataset, val_dataset, test_dataset, tokenizer

def compute_metrics(eval_pred):
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():

    train_dataset, val_dataset, test_dataset = prepare_datasets()
    train_dataset, val_dataset, test_dataset, tokenizer = tokenize_datasets(
        train_dataset, val_dataset, test_dataset
    )
    

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        dropout=0.4,
        attention_dropout=0.2
    )
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        optim="sgd",
        learning_rate=0.001,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=15,
        weight_decay=0.01,
        seed=42,
        logging_dir="./logs",
        logging_steps=10000,  
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=None,
        disable_tqdm=True,
        log_level="error"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EpochLogger()]
    )
    
    print("Fine-tuning")
    
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    trainer.train()
    sys.stdout = original_stdout
    
    print("Training completato!")

    test_results = trainer.evaluate(test_dataset)
    print("\nTest Set :")
    print(f"  Loss:      {test_results['eval_loss']:.6f}")
    print(f"  Accuracy:  {test_results['eval_accuracy']:.6f}")
    print(f"  Precision: {test_results['eval_precision']:.6f}")
    print(f"  Recall:    {test_results['eval_recall']:.6f}")
    print(f"  F1:        {test_results['eval_f1']:.6f}")
    
    output_dir = "./output/sentiment_model_custom"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModello salvato in: {output_dir}")

if __name__ == "__main__":
    main()