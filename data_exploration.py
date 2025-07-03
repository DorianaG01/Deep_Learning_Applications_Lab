from utils import load_rotten_tomatoes

def explore_dataset():
    """Carica ed esplora il dataset Rotten Tomatoes."""
    dataset = load_rotten_tomatoes()
    
    for split_name in ["train", "validation", "test"]:
        split_data = dataset[split_name]
        example = split_data[0]
        sentiment = "Positivo" if example["label"] == 1 else "Negativo"
        print(f"Esempio {split_name}: [{sentiment}] {example['text'][:80]}...")
    
    return dataset

if __name__ == "__main__":
    explore_dataset()