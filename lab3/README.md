# Lab 3: Sentiment Analysis with DistilBERT

Complete sentiment analysis laboratory using DistilBERT on the Rotten Tomatoes dataset.

##  Project Structure

```
lab3/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── utils.py                     # Shared utility functions
├── data_exploration.py          # Dataset exploration
├── feature_extraction.py        # Feature extraction + SVM
├── fine_tuning.py              # DistilBERT fine-tuning
└── translator_storyteller.py   # Image-to-Story generator
```

##  Installation

1. **Clone the repository:**
```bash
git clone https://github.com/DorianaG01/Deep_Learning_Applications_Lab.git
cd Deep_Learning_Applications_Lab/lab3
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## How to Use

### 1. Dataset Exploration
```bash
python data_exploration.py
```
Loads and analyzes the Rotten Tomatoes dataset showing:
- Statistics for each split (train/validation/test)
- Label distribution
- Text examples

### 2. Feature Extraction + SVM
```bash
python feature_extraction.py
```
- Extracts embeddings with DistilBERT ([CLS] representations)
- Trains linear SVM classifier
- Evaluates performance on complete dataset

### 3. DistilBERT Fine-tuning
```bash
python fine_tuning.py
```
- End-to-end DistilBERT fine-tuning
- Training with clean progress table
- Evaluation on validation and test sets

### 4. Image-to-Story Generator
```bash
python translator_storyteller.py
```
- Generates captions from images using BLIP
- Creates stories from captions using TinyStories
- Interactive interface for image input
- Supports URLs and local image files
- Adjustable story length (short/medium/long)

## Dataset

- **Name**: Rotten Tomatoes Movie Reviews
- **Type**: Binary sentiment classification (positive/negative)
- **Size**:
  - Train: 8,530 examples
  - Validation: 1,066 examples  
  - Test: 1,066 examples
- **Balance**: 50% positive, 50% negative

## Models

### DistilBERT (Sentiment Analysis)
- **Version**: `distilbert-base-uncased`
- **Parameters**: 66M (vs 110M of BERT-base)
- **Speed**: ~2x faster than BERT
- **Performance**: ~97% of BERT performance

### BLIP (Image Captioning)
- **Version**: `Salesforce/blip-image-captioning-base`
- **Task**: Image-to-text generation
- **Architecture**: Vision-Language understanding

### TinyStories (Story Generation)
- **Version**: `roneneldan/TinyStories-33M`
- **Parameters**: 33M parameters
- **Task**: Creative text generation
- **Style**: Children's stories

### Configurations
- **SVM**: DistilBERT features + linear classifier
- **Fine-tuning**: End-to-end training with Adam optimizer
- **Story Generation**: Temperature=0.6, top_p=0.9, repetition_penalty=1.2
  
## Training Configuration

```python
# Fine-tuning parameters
learning_rate = 2e-5
batch_size = 16
num_epochs = 3
optimizer = "adam"
max_length = 256
```

##  Troubleshooting

### Common Issues

1. **CUDA out of memory error**
   ```python
   # Reduce batch size in TrainingArguments
   per_device_train_batch_size=8  # instead of 16
   ```

2. **Import Error transformers**
   ```bash
   pip install --upgrade transformers datasets
   ```

3. **Slow performance on CPU**
   - This is normal, training will take longer
   - Consider using smaller subsets for testing

4. **Image loading errors (translator_storyteller.py)**
   ```bash
   # Make sure Pillow and requests are installed
   pip install Pillow requests
   ```

5. **Story generation too repetitive**
   - Increase `repetition_penalty` (default: 1.2)
   - Adjust `temperature` for more/less creativity

### Performance Tips

- **GPU**: Training is ~10x faster
- **Batch Size**: Increase if you have sufficient memory
- **Max Length**: Reduce to 128 for shorter texts
- **Image Size**: Smaller images process faster for story generation

## References

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Rotten Tomatoes Dataset](https://huggingface.co/datasets/rotten_tomatoes)
- [Sentiment Analysis Guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)

## License

Educational project for Deep Learning Applications Lab.
