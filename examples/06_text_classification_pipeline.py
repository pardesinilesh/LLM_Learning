#!/usr/bin/env python3
"""
Example 6: Text Classification Pipeline

Learn how to build a complete text classification pipeline for multiple categories.
Concepts: Multi-class classification, preprocessing, training, evaluation metrics
"""

import sys
from pathlib import Path


def example_1_create_dataset():
    """Create a multi-class classification dataset."""
    print("=" * 60)
    print("EXAMPLE 1: Creating Multi-Class Dataset")
    print("=" * 60)
    
    # Sample dataset with multiple news categories
    dataset = {
        "Sports": [
            "The soccer team won the championship match.",
            "Basketball player scores 50 points in game.",
            "Tennis player advances to final round.",
            "Football team wins playoff game.",
            "Olympics begin with opening ceremony.",
        ],
        "Technology": [
            "New AI model shows impressive results.",
            "Tech company releases smartphone.",
            "AI researchers publish breakthrough paper.",
            "Software update improves performance.",
            "Neural networks advance machine learning.",
        ],
        "Politics": [
            "Government announces new policy.",
            "Political candidate wins election.",
            "Parliament debates bill.",
            "President gives speech.",
            "Congress passes legislation.",
        ],
        "Entertainment": [
            "New movie breaks box office records.",
            "Celebrity wins award.",
            "Music festival announces lineup.",
            "Actor stars in new film.",
            "Concert sells out in minutes.",
        ],
    }
    
    print("\nDataset structure: 4 categories, 5 examples each\n")
    
    for category, texts in dataset.items():
        print(f"{category}:")
        for i, text in enumerate(texts, 1):
            print(f"  {i}. {text}")
        print()
    
    # Flatten data
    texts = []
    labels = []
    label_to_id = {}
    
    for label_id, (category, category_texts) in enumerate(dataset.items()):
        label_to_id[category] = label_id
        texts.extend(category_texts)
        labels.extend([label_id] * len(category_texts))
    
    print(f"Total examples: {len(texts)}")
    print(f"Categories: {list(label_to_id.keys())}")
    print(f"Label mapping: {label_to_id}\n")
    
    return texts, labels, label_to_id


def example_2_preprocessing_pipeline(texts, labels):
    """Show text preprocessing steps."""
    print("=" * 60)
    print("EXAMPLE 2: Text Preprocessing Pipeline")
    print("=" * 60)
    
    print("\nText preprocessing improves model performance.\n")
    
    # Original text
    original_text = "The AI Model Shows IMPRESSIVE Results!!!"
    print(f"1. Original text:")
    print(f"   '{original_text}'\n")
    
    # Lowercase
    lowercased = original_text.lower()
    print(f"2. Lowercasing:")
    print(f"   '{lowercased}'\n")
    
    # Remove punctuation
    import string
    no_punct = lowercased.translate(str.maketrans('', '', string.punctuation))
    print(f"3. Remove punctuation:")
    print(f"   '{no_punct}'\n")
    
    # Tokenization
    tokens = no_punct.split()
    print(f"4. Tokenization:")
    print(f"   {tokens}\n")
    
    # Remove common words (stopwords)
    stopwords = {"the", "a", "an", "and", "or", "is", "in", "to"}
    filtered = [t for t in tokens if t not in stopwords]
    print(f"5. Remove stopwords:")
    print(f"   {filtered}\n")
    
    print("Full preprocessing function:\n")
    
    def preprocess_text(text):
        """Complete preprocessing pipeline."""
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = text.split()
        # Remove stopwords
        stopwords = {"the", "a", "an", "and", "or", "is", "in", "to", "at", "of"}
        tokens = [t for t in tokens if t not in stopwords]
        return ' '.join(tokens)
    
    # Apply to samples
    print("Applied to sample texts:\n")
    for i, text in enumerate(texts[:3], 1):
        processed = preprocess_text(text)
        print(f"{i}. Original:   '{text}'")
        print(f"   Processed:  '{processed}\n")


def example_3_tokenization_and_encoding(texts, labels, label_to_id):
    """Tokenize texts and prepare for model training."""
    print("=" * 60)
    print("EXAMPLE 3: Tokenization and Encoding")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers library not installed.")
        return
    
    print("\n1. Loading tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    except Exception as e:
        print(f"Note: Could not load tokenizer: {e}")
        return
    
    print("   ✓ Tokenizer loaded\n")
    
    # Tokenize samples
    print("2. Tokenizing sample texts:\n")
    
    for i, text in enumerate(texts[:3], 1):
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        
        print(f"Text {i}: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Length: {len(token_ids)} tokens\n")
    
    print("3. Batch encoding:\n")
    
    # Encode full batch
    encodings = tokenizer(
        texts[:5],
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    print(f"   Input IDs shape: {encodings['input_ids'].shape}")
    print(f"   Attention mask shape: {encodings['attention_mask'].shape}")
    print(f"   Token type IDs shape: {encodings['token_type_ids'].shape}\n")
    
    # Show actual encoded values
    print("4. Sample encoded output:")
    print(f"   Input IDs (first 5 tokens): {encodings['input_ids'][0][:5]}")
    print(f"   Attention mask (first 5): {encodings['attention_mask'][0][:5]}\n")


def example_4_training_and_evaluation(texts, labels, label_to_id):
    """Set up training and evaluation workflow."""
    print("=" * 60)
    print("EXAMPLE 4: Training and Evaluation")
    print("=" * 60)
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        import numpy as np
    except ImportError:
        print("ERROR: scikit-learn not installed.")
        print("Run: pip install scikit-learn")
        return
    
    print("\n1. Splitting data into train and test sets...")
    
    # Split data (70% train, 30% test)
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"   Training set: {len(texts_train)} examples ({len(texts_train)/len(texts)*100:.1f}%)")
    print(f"   Test set: {len(texts_test)} examples ({len(texts_test)/len(texts)*100:.1f}%)\n")
    
    # Show class distribution
    print("2. Class distribution:\n")
    
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    for label_id in sorted(id_to_label.keys()):
        train_count = sum(1 for l in labels_train if l == label_id)
        test_count = sum(1 for l in labels_test if l == label_id)
        
        print(f"   {id_to_label[label_id]:15} Train: {train_count:2d}  Test: {test_count:2d}")
    
    print("\n3. Evaluation metrics explained:\n")
    
    print("""
    Accuracy: Percentage of correct predictions
    
    Precision: Of predicted positive, how many are actually positive?
    Formula: TP / (TP + FP)
    
    Recall: Of actual positive, how many are predicted correct?
    Formula: TP / (TP + FN)
    
    F1-Score: Harmonic mean of precision and recall
    Formula: 2 * (Precision * Recall) / (Precision + Recall)
    
    TP (True Positive): Correct positive prediction
    FP (False Positive): Wrong positive prediction
    FN (False Negative): Missed positive prediction
    """)
    
    # Dummy predictions for demo
    print("4. Example evaluation (with dummy predictions):\n")
    
    dummy_predictions = [id_to_label[l][:4] if isinstance(id_to_label[l], str) else l 
                        for l in labels_test]
    dummy_predictions_encoded = [label_to_id[id_to_label[l]] 
                                for l in labels_test]
    
    accuracy = accuracy_score(labels_test, dummy_predictions_encoded)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_test, dummy_predictions_encoded, average='weighted'
    )
    
    print(f"   Accuracy:  {accuracy:.2%}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall:    {recall:.2%}")
    print(f"   F1-Score:  {f1:.2%}\n")
    
    print("5. Per-category metrics:\n")
    
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels_test, dummy_predictions_encoded, average=None
    )
    
    for label_id, category in id_to_label.items():
        print(f"   {category}:")
        print(f"      Precision: {precision_per_class[label_id]:.2%}")
        print(f"      Recall:    {recall_per_class[label_id]:.2%}")
        print(f"      F1-Score:  {f1_per_class[label_id]:.2%}")


def example_5_confusion_matrix_analysis():
    """Explain confusion matrix for understanding errors."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Confusion Matrix Analysis")
    print("=" * 60)
    
    print("""
A confusion matrix shows what the model predicted vs actual labels:

                 Predicted
                Sports  Tech   Pol  Ent
        Sports  |  4     0     0    1  |
Actual  Tech    |  0     5     0    0  |
        Pol     |  1     0     3    1  |
        Ent     |  0     0     0    5  |

Reading the matrix:
- Diagonal (✓): Correct predictions
- Off-diagonal (✗): Errors/confusions

Example insights:
- Model confuses 1 Sports with Entertainment
- 1 Politics with Sports
- All Tech and Entertainment predictions are correct

Common confusions help identify:
1. Similar categories that need better data
2. Ambiguous examples that need labeling review
3. Categories that need more training data
    """)


def example_6_hyperparameter_tuning():
    """Explain hyperparameter tuning for classification."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Hyperparameter Tuning")
    print("=" * 60)
    
    print("""
Important hyperparameters for text classification:

1. Learning Rate (most important!)
   - Too high: Model diverges, performance drops
   - Too low: Training is very slow
   - Typical range: 1e-5 to 1e-3
   - Default for fine-tuning: 5e-5

2. Batch Size
   - Larger: Faster training, more stable gradients
   - Smaller: Better generalization, less memory
   - Typical range: 8 to 64
   - Trade-off: Speed vs accuracy

3. Number of Epochs
   - Too few: Underfitting (model too simple)
   - Too many: Overfitting (memorizes training data)
   - Typical: 3-5 epochs for fine-tuning

4. Max Sequence Length
   - Affects speed and memory usage
   - Longer sequences = more computation
   - Typical: 128-512 tokens

5. Warmup Steps
   - Gradually increase learning rate from 0
   - Helps training stability
   - Typical: 500-2000 steps

Recommended starting point:
  - Learning rate: 2e-5
  - Batch size: 16
  - Epochs: 3
  - Max length: 128
  - Warmup: 500 steps

Tuning strategy:
1. Start with defaults
2. Train and evaluate
3. If overfitting: reduce epochs or batch size
4. If underfitting: increase epochs or learning rate
5. Iterate until satisfied
    """)


def example_7_production_pipeline():
    """Show complete production pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Production Classification Pipeline")
    print("=" * 60)
    
    try:
        from transformers import pipeline
    except ImportError:
        print("ERROR: transformers library not installed.")
        return
    
    print("\n1. Loading pre-trained classifier...\n")
    
    try:
        # Use a pre-trained zero-shot classifier
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
    except Exception as e:
        print(f"Note: Could not load model: {e}")
        return
    
    print("   ✓ Classifier loaded\n")
    
    # Define categories
    categories = ["Sports", "Technology", "Politics", "Entertainment"]
    
    # Test texts
    test_texts = [
        "The soccer team wins the championship!",
        "New AI breakthrough announced.",
        "Government passes new law.",
        "Famous actor wins award.",
    ]
    
    print("2. Classifying test examples:\n")
    
    for text in test_texts:
        result = classifier(text, categories)
        predicted_category = result['labels'][0]
        confidence = result['scores'][0]
        
        print(f"Text: '{text}'")
        print(f"  → Predicted: {predicted_category} ({confidence:.1%})")
        print(f"  Scores: {dict(zip(result['labels'], [f'{s:.1%}' for s in result['scores']]))}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TEXT CLASSIFICATION PIPELINE EXAMPLES")
    print("=" * 60)
    
    # Example 1: Create dataset
    texts, labels, label_to_id = example_1_create_dataset()
    
    # Example 2: Preprocessing
    example_2_preprocessing_pipeline(texts, labels)
    
    # Example 3: Tokenization
    example_3_tokenization_and_encoding(texts, labels, label_to_id)
    
    # Example 4: Training and evaluation
    example_4_training_and_evaluation(texts, labels, label_to_id)
    
    # Example 5: Confusion matrix
    example_5_confusion_matrix_analysis()
    
    # Example 6: Hyperparameter tuning
    example_6_hyperparameter_tuning()
    
    # Example 7: Production pipeline
    example_7_production_pipeline()
    
    print("=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print("""
KEY TAKEAWAYS:

1. Multi-class classification: Predict one of many categories
2. Preprocessing: Clean data before training
3. Train/test split: Evaluate on unseen data
4. Evaluation metrics: Accuracy, precision, recall, F1
5. Hyperparameter tuning: Optimize for your use case
6. Production: Use pre-trained models for quick deployment

NEXT STEPS:
- Collect real data for your domain
- Train a custom classifier on your categories
- Monitor performance metrics
- Iterate to improve results
    """)


if __name__ == "__main__":
    main()
