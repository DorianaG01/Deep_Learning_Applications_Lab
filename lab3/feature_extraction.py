"""
Laboratorio di Sentiment Analysis - Parte 2: Estrazione Features con DistilBERT
"""

from utils import load_rotten_tomatoes, get_dataset_splits
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import torch
import numpy as np

def extract_features(texts, tokenizer, model, device, batch_size=8):
    """Estrae embeddings [CLS] dai testi."""
    model.eval()
    all_embeddings = []
    
    print(f"Estraendo features da {len(texts)} testi...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(batch_texts, padding=True, truncation=True, 
                          max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

def main():

    dataset = load_rotten_tomatoes()
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_dataset_splits(dataset)
    

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Usando: {device}")
    
    train_features = extract_features(train_texts, tokenizer, model, device)
    test_features = extract_features(test_texts, tokenizer, model, device)
    
    print("Addestrando SVM...")
    classifier = SVC(kernel="linear", C=1.0)
    classifier.fit(train_features, train_labels)
    
    test_predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, test_predictions)
    
    print(f"\nRisultati SVM:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nReport dettagliato:")
    print(classification_report(test_labels, test_predictions, 
                              target_names=["Negativo", "Positivo"]))
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

if __name__ == "__main__":
    main()