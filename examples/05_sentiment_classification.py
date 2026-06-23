#!/usr/bin/env python3
"""
Example 5: Sentiment Classification

Learn how to train and evaluate a model on a text classification task.
Concepts: Fine-tuning, training loops, evaluation metrics, binary classification
"""

import sys
import csv
from pathlib import Path


def example_1_load_and_explore_data():
    """Load and explore sentiment dataset."""
    print("=" * 60)
    print("EXAMPLE 1: Loading Sentiment Dataset")
    print("=" * 60)
    
    # Load dataset
    data_path = Path(__file__).parent.parent / "data" / "sentiment_examples.csv"
    
    texts = []
    labels = []
    
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("Make sure sentiment_examples.csv exists in data/ folder")
        return None, None
    
    print(f"\nLoading data from: {data_path}\n")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            labels.append(int(row['label']))
    
    print(f"✓ Loaded {len(texts)} examples\n")
    
    # Explore dataset
    positive = sum(1 for l in labels if l == 1)
    negative = sum(1 for l in labels if l == 0)
    
    print("Dataset Statistics:")
    print(f"  Positive examples: {positive}")
    print(f"  Negative examples: {negative}")
    print(f"  Total examples: {len(texts)}")
    print(f"  Class balance: {positive/len(labels)*100:.1f}% positive\n")
    
    # Show examples
    print("Sample positive examples:")
    for i, text in enumerate([t for t, l in zip(texts, labels) if l == 1][:3]):
        print(f"  {i+1}. {text}")
    
    print("\nSample negative examples:")
    for i, text in enumerate([t for t, l in zip(texts, labels) if l == 0][:3]):
        print(f"  {i+1}. {text}")
    
    return texts, labels


def example_2_simple_sentiment_classification(texts, labels):
    """Use zero-shot classification, no training required."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Zero-Shot Sentiment Classification")
    print("=" * 60)
    
    try:
        from transformers import pipeline
    except ImportError:
        print("ERROR: transformers library not installed.")
        print("Run: pip install -r requirements.txt")
        return
    
    print("\n1. Loading zero-shot classification pipeline...")
    print("   (This means no training - the model already knows about sentiment)\n")
    
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU
        )
    except Exception as e:
        print(f"Note: Could not load model: {e}")
        print("Make sure you have internet connection.")
        return
    
    print("   ✓ Model loaded!\n")
    
    # Define candidate labels
    candidate_labels = ["positive", "negative"]
    
    # Classify a few examples
    sample_texts = texts[:5] if texts else ["This is great!", "This is terrible"]
    sample_labels = labels[:5] if labels else [1, 0]
    
    print("2. Classifying examples:\n")
    
    correct = 0
    for text, true_label in zip(sample_texts, sample_labels):
        result = classifier(text, candidate_labels)
        predicted_label = 1 if result['labels'][0] == 'positive' else 0
        
        # Check if correct
        is_correct = "✓" if predicted_label == true_label else "✗"
        true_sentiment = "positive" if true_label == 1 else "negative"
        pred_sentiment = result['labels'][0]
        
        print(f"{is_correct} Text: '{text}'")
        print(f"   True: {true_sentiment} | Predicted: {pred_sentiment}")
        print(f"   Confidence: {result['scores'][0]:.2%}\n")
        
        if predicted_label == true_label:
            correct += 1
    
    accuracy = correct / len(sample_texts) * 100 if sample_texts else 0
    print(f"Accuracy on sample: {accuracy:.1f}%")


def example_3_fine_tuning_setup(texts, labels):
    """Show how to set up fine-tuning (training loop outline)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Fine-Tuning Setup (Conceptual)")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from torch.utils.data import Dataset, DataLoader
        import torch
    except ImportError:
        print("ERROR: Required libraries not installed.")
        print("Run: pip install -r requirements.txt")
        return
    
    if not texts or not labels:
        print("ERROR: No data available")
        return
    
    print("\n1. Loading pre-trained model and tokenizer...")
    
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # Binary classification: positive/negative
        )
    except Exception as e:
        print(f"Note: Could not load model: {e}")
        return
    
    print("   ✓ Model loaded!\n")
    
    print("2. Setting up dataset class...")
    
    class SentimentDataset(Dataset):
        """Custom dataset for sentiment classification."""
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(label)
            }
    
    print("   ✓ Dataset class created\n")
    
    # Create dataset
    print("3. Creating dataset from examples...")
    dataset = SentimentDataset(texts, labels, tokenizer)
    print(f"   ✓ Dataset size: {len(dataset)}\n")
    
    # Show batch example
    print("4. Example tokenized batch:")
    sample_batch = dataset[0]
    print(f"   Input IDs shape: {sample_batch['input_ids'].shape}")
    print(f"   Attention mask shape: {sample_batch['attention_mask'].shape}")
    print(f"   Label: {sample_batch['labels'].item()}\n")
    
    # Create dataloader
    print("5. Creating DataLoader for batching...")
    train_dataloader = DataLoader(dataset, batch_size=4)
    print(f"   ✓ DataLoader created")
    print(f"   Number of batches: {len(train_dataloader)}\n")
    
    print("6. Training loop structure (pseudo-code):\n")
    
    print("""
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch}: Loss = {avg_loss:.4f}')
    
    # Save model
    model.save_pretrained('my_sentiment_model')
    """)


def example_4_practical_inference():
    """Use a pre-trained sentiment model for inference."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Practical Sentiment Inference")
    print("=" * 60)
    
    try:
        from transformers import pipeline
    except ImportError:
        print("ERROR: transformers library not installed.")
        print("Run: pip install -r requirements.txt")
        return
    
    print("\n1. Loading sentiment analysis pipeline...")
    
    try:
        # Using distilBERT fine-tuned on sentiment
        sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU
        )
    except Exception as e:
        print(f"Note: Could not load model: {e}")
        print("Make sure you have internet connection.")
        return
    
    print("   ✓ Model loaded!\n")
    
    # Test sentences
    test_sentences = [
        "I absolutely love this product!",
        "This is the worst experience ever.",
        "It's okay, nothing special.",
        "Amazing quality and fast shipping!",
        "Terrible customer service.",
    ]
    
    print("2. Analyzing sentiment of test sentences:\n")
    
    for sentence in test_sentences:
        result = sentiment(sentence)[0]
        label = result['label']
        score = result['score']
        
        emoji = "😊" if label == 'POSITIVE' else "😞"
        print(f"{emoji} '{sentence}'")
        print(f"    → {label} ({score:.2%})\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SENTIMENT CLASSIFICATION EXAMPLES")
    print("=" * 60)
    
    # Example 1: Load and explore
    texts, labels = example_1_load_and_explore_data()
    
    if texts is None:
        print("\nSkipping remaining examples due to data loading error.")
        return
    
    # Example 2: Zero-shot classification
    example_2_simple_sentiment_classification(texts, labels)
    
    # Example 3: Fine-tuning setup
    example_3_fine_tuning_setup(texts, labels)
    
    # Example 4: Practical inference
    example_4_practical_inference()
    
    print("\n" + "=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print("""
KEY TAKEAWAYS:

1. Zero-shot classification: Use models without any training
2. Fine-tuning: Adapt pre-trained models to your specific task
3. Evaluation: Measure accuracy, precision, recall on test data
4. Inference: Use trained models to make predictions

NEXT STEPS:
- Collect more data for better results
- Use different model architectures (BERT, RoBERTa, etc.)
- Experiment with hyperparameters (learning rate, batch size)
- Add validation set to avoid overfitting
    """)


if __name__ == "__main__":
    main()
