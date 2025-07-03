from datasets import load_dataset
from collections import Counter

def load_rotten_tomatoes():
    dataset = load_dataset("rotten_tomatoes")
   
    for split_name in ["train", "validation", "test"]:
        split_data = dataset[split_name]
        label_counts = Counter(split_data["label"])
        print(f"  {split_name}: {len(split_data)} esempi - {dict(label_counts)}")
    
    return dataset

def get_dataset_splits(dataset):
    return (
        dataset["train"]["text"], dataset["train"]["label"],
        dataset["validation"]["text"], dataset["validation"]["label"], 
        dataset["test"]["text"], dataset["test"]["label"]
    )