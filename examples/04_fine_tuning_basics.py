#!/usr/bin/env python3
"""
Example 4: Fine-Tuning Basics

Learn how to adapt a pre-trained model to a specific task.
Concepts: Transfer learning, fine-tuning, task adaptation
"""

import sys


def example_1_understanding_finetuning():
    """Explain fine-tuning conceptually."""
    print("=" * 60)
    print("EXAMPLE 1: What is Fine-Tuning?")
    print("=" * 60)
    
    print("""
ANALOGY: Learning to Drive

1. PRE-TRAINING (What the model already knows)
   Like learning the basics of a car in driving school:
   - How to steer
   - How to use pedals
   - Basic traffic rules
   Takes years of learning patterns
   
   LLM Pre-training:
   - Learned from 100+ billion words
   - Knows grammar, facts, reasoning
   - General language understanding
   
2. FINE-TUNING (Specializing for a task)
   Like learning to drive a taxi in NYC:
   - You keep all your driving skills
   - But adapt them for your specific job
   - Learn street names, traffic patterns
   - Takes much less time!
   
   LLM Fine-tuning:
   - Start with pre-trained model
   - Train on task-specific data (sentiment, classification, etc.)
   - Keep general knowledge, add task expertise
   - Takes hours instead of months
   
WHY IS THIS POWERFUL?

→ Much faster training
→ Needs less data (100-1000 examples vs millions)
→ Better performance than training from scratch
→ Can customize general models for YOUR needs
""")


def example_2_simple_finetuning():
    """Demonstrate simple fine-tuning."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Text Classification Fine-Tuning")
    print("=" * 60)
    
    try:
        from transformers import (
            AutoTokenizer, 
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments
        )
        import torch
    except ImportError:
        print("ERROR: Required libraries not installed.")
        print("Run: pip install -r requirements.txt")
        return
    
    print("\n1. Creating a simple dataset...")
    
    # Simple sentiment dataset (in production, load from file)
    train_data = {
        "text": [
            "This product is amazing!",
            "I love this movie",
            "Excellent service",
            "Terrible quality",
            "I hate waiting",
            "Worst experience ever",
            "This is great",
            "Not good at all",
        ],
        "label": [1, 1, 1, 0, 0, 0, 1, 0]  # 1=positive, 0=negative
    }
    
    print(f"   Dataset size: {len(train_data['text'])} samples")
    print("   Samples:")
    for text, label in zip(train_data['text'][:3], train_data['label'][:3]):
        label_name = "positive" if label == 1 else "negative"
        print(f"   - '{text}' → {label_name}")
    
    print("\n2. Loading pre-trained model...")
    
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2  # binary classification
        )
    except Exception as e:
        print(f"Note: Could not load model: {e}")
        return
    
    print(f"   ✓ Loaded {model_name}")
    print(f"   Number of parameters: ~110M")
    print("   (We'll train only the last few layers)")
    
    print("\n3. Tokenizing data...")
    
    # Tokenize all texts
    encodings = tokenizer(
        train_data['text'],
        truncation=True,
        padding=True,
        max_length=128
    )
    
    print(f"   ✓ Tokenized {len(encodings['input_ids'])} samples")
    print(f"   Example token count: {len(encodings['input_ids'][0])} tokens")
    
    print("\n4. Creating dataset...")
    
    from torch.utils.data import Dataset
    
    class SentimentDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
    
    dataset = SentimentDataset(encodings, train_data['label'])
    print(f"   ✓ Created dataset with {len(dataset)} samples")
    
    print("\n5. Setting up training...")
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
    )
    
    print("""
   Training configuration:
   - Learning rate: 2e-5 (small, to preserve pre-training)
   - Batch size: 8
   - Epochs: 3
   - Optimizer: AdamW (specialized for fine-tuning)
    """)
    
    print("\n6. Training the model...")
    print("   (In a real scenario, this would train for a few minutes)")
    print("   (For this demo, we'll show the training setup)\n")
    
    # Don't actually train to save time
    print("   ✓ Model setup complete!")
    print("   ✓ Ready to train!")
    
    print("\n7. Expected training progress:")
    print("""
   Epoch 1/3
     Step 1/4: Loss=0.456
     Step 2/4: Loss=0.234
     Step 3/4: Loss=0.123
     Step 4/4: Loss=0.089
     Val accuracy: 87%
   
   Epoch 2/3
     Step 1/4: Loss=0.076
     Step 2/4: Loss=0.052
     ...
     Val accuracy: 95%
   
   Epoch 3/3
     ...
     Val accuracy: 96%
    """)


def example_3_when_to_finetune():
    """Explain when fine-tuning is useful."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: When to Use Fine-Tuning")
    print("=" * 60)
    
    print("""
✅ GOOD USE CASES FOR FINE-TUNING:

1. CLASSIFICATION TASKS
   ├─ Sentiment analysis
   ├─ Spam detection
   ├─ Topic classification
   └─ Intent detection
   
   Why: Models learn to be sensitive to task-specific patterns
   Data needed: 100-1000 examples

2. DOMAIN-SPECIFIC TASKS
   ├─ Medical text classification
   ├─ Legal document categorization
   ├─ Technical support issues
   └─ Scientific paper analysis
   
   Why: Pre-training + domain data = domain expertise
   Data needed: 500-5000 examples

3. PARAPHRASING & SUMMARIZATION
   ├─ Paraphrase generation
   ├─ Document summarization
   └─ Text simplification
   
   Why: Models learn task-specific patterns
   Data needed: 500-5000 examples

❌ CASES WHERE FINE-TUNING MIGHT NOT HELP:

1. ❌ Very simple tasks
   Example: Classifying text as "yes" or "no"
   Better: Use prompt engineering with zero-shot
   
2. ❌ Few examples (< 50)
   Problem: Risk overfitting
   Better: Few-shot prompting or larger models
   
3. ❌ Data not representative
   Problem: Model learns spurious patterns
   Better: Collect better data first
   
4. ❌ Tasks requiring reasoning
   Example: Complex multi-step reasoning
   Better: Use larger pre-trained models with prompting

⚠️  COMMON MISTAKES:

1. Training for too many epochs
   → Model forgets pre-trained knowledge (catastrophic forgetting)
   → Solution: Use low learning rate + early stopping
   
2. Using too high learning rate
   → Weights change too much, lose pre-training
   → Solution: Use learning rate 1e-5 to 5e-5
   
3. Training without validation
   → Don't know if model is improving
   → Solution: Use train/val split + monitoring
   
4. Fine-tuning the whole model
   → Slower and needs more data
   → Better: Fine-tune only last 2-3 layers
""")


def example_4_parameter_efficiency():
    """Show efficient fine-tuning methods."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Efficient Fine-Tuning Methods")
    print("=" * 60)
    
    print("""
Modern fine-tuning is more efficient than training all parameters!

1. STANDARD FINE-TUNING
   ├─ Fine-tune all parameters
   ├─ Training time: Minutes to hours
   ├─ Memory: High
   └─ Performance: Excellent
   
   Use when: Good GPU available, task-critical
   
2. PARAMETER-EFFICIENT FINE-TUNING

   a) Layer Freezing
      ├─ Freeze early layers (use pre-training)
      ├─ Only train last 2-3 layers
      ├─ Training time: 10x faster
      └─ Memory: 10x lower
      
      Use when: Limited compute
      
   b) LoRA (Low-Rank Adaptation)
      ├─ Add small trainable matrices
      ├─ Much smaller than full fine-tuning
      ├─ Training time: 2-3x faster
      └─ Memory: 90% lower
      
      Use when: Want efficiency + performance
      
   c) Adapter Modules
      ├─ Add small modules between layers
      ├─ Fine-tune only adapters
      ├─ Training time: 2-3x faster
      └─ Memory: 50-90% lower
      
      Use when: Need modularity

COMPARISON TABLE:

Method                    | Time    | Memory  | Performance
Standard Fine-tuning      | 1x      | 1x      | ★★★★★
Layer Freezing            | 0.1x    | 0.1x    | ★★★★☆
LoRA                      | 0.3x    | 0.1x    | ★★★★★
Adapter Modules           | 0.4x    | 0.2x    | ★★★★☆
Prompt Tuning            | 0.05x   | 0.05x   | ★★★☆☆

RECOMMENDATION:
→ Start with LoRA for best balance
→ Use prompt tuning for quick experiments
→ Use full fine-tuning when performance is critical
""")


def example_5_practical_workflow():
    """Show practical fine-tuning workflow."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Practical Fine-Tuning Workflow")
    print("=" * 60)
    
    print("""
STEP-BY-STEP WORKFLOW:

1. DATA PREPARATION
   ├─ Collect task-specific data
   ├─ Split: train (80%) / validation (20%)
   ├─ Check for quality and balance
   └─ Ensure representative sampling
   
   Time: 1-2 days
   
2. BASELINE
   ├─ Test pre-trained model as-is
   ├─ Evaluate on test set
   └─ This becomes your baseline
   
   Time: < 1 hour
   
3. INITIAL FINE-TUNE
   ├─ Set up training pipeline
   ├─ Use small subset first (test run)
   ├─ Check for errors
   └─ Verify GPU usage is reasonable
   
   Time: 1-3 hours
   
4. HYPERPARAMETER TUNING
   ├─ Try different learning rates (2e-5, 5e-5, 1e-4)
   ├─ Different batch sizes (8, 16, 32)
   ├─ Different architectures
   └─ Use validation for selection
   
   Time: 1-2 days
   
5. FINAL TRAINING
   ├─ Train with best hyperparameters
   ├─ Train for full duration
   ├─ Monitor validation metrics
   └─ Save best checkpoint
   
   Time: 1-4 hours
   
6. EVALUATION
   ├─ Test on held-out test set
   ├─ Compare to baseline
   ├─ Analyze errors
   └─ Document results
   
   Time: 1-2 hours

TOTAL TIME: 3-5 days for production-quality model

KEY METRICS TO TRACK:
✓ Training loss (should decrease)
✓ Validation loss (should decrease, but less than training)
✓ Validation accuracy (should increase)
✓ GPU memory usage (should be stable)
✓ Training time per epoch (should be consistent)
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LEARNING OBJECTIVE")
    print("=" * 60)
    print("""
In this example, you'll learn:
1. What fine-tuning is and why it's powerful
2. How to set up a fine-tuning pipeline
3. When to use fine-tuning vs other methods
4. Practical training considerations
5. Tips for successful adaptation

Read docs/04_practical_guide.md for deeper insights!
    """)
    
    example_1_understanding_finetuning()
    example_2_simple_finetuning()
    example_3_when_to_finetune()
    example_4_parameter_efficiency()
    example_5_practical_workflow()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
✓ Fine-tuning adapts pre-trained models to specific tasks
✓ Much faster and needs less data than training from scratch
✓ Use low learning rates to preserve pre-trained knowledge
✓ Monitor validation performance to avoid overfitting
✓ Consider parameter-efficient methods for production
✓ Perfect for domain-specific and classification tasks

NEXT STEPS:
→ Implement your own fine-tuning pipeline
→ Test on your own data
→ Try different architectures and methods
→ Deploy your fine-tuned model!
    """)
