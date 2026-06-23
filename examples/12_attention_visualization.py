#!/usr/bin/env python3
"""
Example 12: Attention Visualization

Learn how to visualize and interpret attention mechanisms in transformers.
Concepts: Self-attention, multi-head attention, attention weights, head importance
"""

import sys
from pathlib import Path


def example_1_attention_basics():
    """Explain attention mechanism concepts."""
    print("=" * 60)
    print("EXAMPLE 1: Attention Mechanism Fundamentals")
    print("=" * 60)
    
    print("""
WHAT IS ATTENTION?

Attention allows the model to focus on relevant parts of input.

ANALOGY: Reading a Sentence
"The movie was great, but the plot was confusing."

If asked "What was confusing?", your attention focuses on:
  The → movie → great → plot → confusing ✓

You ignore irrelevant words and focus on key relationships.

THE ATTENTION MECHANISM:

Input: "The movie was great but plot was confusing"
             ↓
1. CREATE REPRESENTATIONS
   - Query (Q): What am I looking for?
   - Key (K): What information do I contain?
   - Value (V): What information should I return?

2. COMPUTE SIMILARITY
   - Compare Query with all Keys
   - Higher score = more relevant
   - Formula: Attention = softmax(Q·K^T / √d_k)

3. USE WEIGHTS
   - Weighted sum of Values
   - High attention weight → use that information
   - Low attention weight → ignore

4. OUTPUT
   - Weighted combination of all Values


MATHEMATICAL FORMULA:

Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q = Query matrix
- K = Key matrix
- V = Value matrix
- d_k = dimension of keys
- √d_k = scaling factor (prevents vanishing gradients)


SELF-ATTENTION vs CROSS-ATTENTION:

Self-Attention:
- Query, Key, Value all from same source
- Word attends to all other words
- Used in transformers

Cross-Attention:
- Query from one source, Key/Value from another
- Used in encoder-decoder models
- Example: Machine translation
    """)


def example_2_multi_head_attention():
    """Explain multi-head attention."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multi-Head Attention")
    print("=" * 60)
    
    print("""
WHAT IS MULTI-HEAD ATTENTION?

Instead of one attention mechanism, use multiple in parallel!

Traditional Attention (Single Head):
  All tokens → Attention → Single output

Multi-Head Attention (Multiple Heads):
  All tokens → Head 1 ──┐
            → Head 2 ──┼→ Concatenate → Output
            → Head 8 ──┘

BENEFITS:

1. Different Perspectives
   Head 1: Focus on syntax (grammar relationships)
   Head 2: Focus on semantics (meaning relationships)
   Head 3: Focus on long-range dependencies
   Head 8: Focus on specific token patterns

2. Richer Representations
   Each head learns different patterns
   Combined output is more informative

3. Robustness
   Some heads might fail, others still work
   Distributed computation

EXAMPLE WITH SENTENCE:

Sentence: "The cat sat on the mat"

Head 1 (Grammar):
  "The" → attends to "cat" (article-noun relationship)
  "sat" → attends to "cat" (subject-verb relationship)

Head 2 (Semantics):
  "cat" → attends to "mat" (location indication)
  "sat" → attends to "mat" (action location)

Head 3 (Position):
  "mat" → attends to nearby words (distance pattern)

Combined: Rich understanding of sentence structure


ATTENTION HEAD PATTERNS:

Typical patterns in trained models:

1. Position Attention
   Pay attention based on word position
   Common in early layers

2. Syntactic Attention
   Capture grammar: subject-verb, noun-adjective
   Common in middle layers

3. Semantic Attention
   Focus on related meanings
   Common in later layers

4. Token Copying
   Directly copy input token
   Used for rare tokens

5. Broad Attention
   Attend equally to all tokens
   Used for general context
    """)


def example_3_visualize_attention(text=""):
    """Visualize attention patterns."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Visualizing Attention Weights")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        import numpy as np
    except ImportError:
        print("ERROR: Required libraries not installed.")
        return
    
    print("\n1. Loading model with attention output...")
    
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
    except Exception as e:
        print(f"Note: Could not load model: {e}")
        return
    
    print("   ✓ Model loaded\n")
    
    # Sample text
    if not text:
        text = "The quick brown fox jumps over the lazy dog"
    
    print(f"2. Analyzing text: '{text}'\n")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    print(f"   Tokens: {tokens}\n")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get attention
    attention = outputs[-1]  # Last tuple is attentions
    
    print(f"3. Attention structure:")
    print(f"   Number of layers: {len(attention)}")
    print(f"   Number of heads: {attention[0].shape[1]}")
    print(f"   Sequence length: {attention[0].shape[2]}\n")
    
    # Analyze first layer, first head
    layer_idx = 0
    head_idx = 0
    
    attention_weights = attention[layer_idx][0, head_idx].numpy()
    
    print(f"4. Attention map (Layer {layer_idx}, Head {head_idx}):")
    print("   (attention from each token to all tokens)\n")
    
    print("        ", end="")
    for token in tokens:
        print(f"{token:8}", end="")
    print()
    
    for i, token in enumerate(tokens):
        print(f"{token:6}", end="")
        for j in range(len(tokens)):
            weight = attention_weights[i, j]
            # Represent as star intensity
            stars = int(weight * 5)
            print(f"{'*' * stars:8}", end="")
        print()
    
    print("\nInterpretation:")
    print("- More stars = more attention")
    print("- * means token attends to that position")
    print("- [CLS] and [SEP] are special tokens\n")


def example_4_attention_patterns():
    """Explain common attention patterns."""
    print("=" * 60)
    print("EXAMPLE 4: Common Attention Patterns")
    print("=" * 60)
    
    patterns = {
        "Position Pattern": {
            "Description": "Attention based on word position",
            "Example": """
            Attends to nearby tokens (local context)
            Position-based masking creates diagonal pattern
            
            [CLS]  The  quick  brown  fox
            [CLS]   ▢     -      -      -
            The     ▢     ▢      -      -
            quick   -     ▢      ▢      -
            brown   -     -      ▢      ▢
            fox     -     -      -      ▢
            
            ▢ = attention, - = no attention
            """,
            "When Used": "Early transformer layers"
        },
        
        "Attending to [CLS]": {
            "Description": "All tokens attend to [CLS] token",
            "Example": """
            [CLS] serves as global context
            
            [CLS]  The  quick  brown
            [CLS]   ▢     -      -
            The     ▢     ▢      -
            quick   ▢     -      ▢
            brown   ▢     -      -
            
            All words look at [CLS] for global info
            """,
            "When Used": "Pooling for classification"
        },
        
        "Query-Key Matching": {
            "Description": "Attend to semantically similar tokens",
            "Example": """
            Adjectives attend to nouns they modify
            
            Example: "good movie"
            good → attends to movie
            movie → attends to good
            
            Related concepts have high attention
            """,
            "When Used": "Middle to late layers"
        },
        
        "Copy Pattern": {
            "Description": "Attend only to self token",
            "Example": """
            Token only attends to itself
            
            Word     Word  itself?
            dog  [0.01, 0.97, 0.02]
            cat  [0.02, 0.96, 0.02]
            
            Mostly attends to self
            """,
            "When Used": "Copying rare tokens"
        },
    }
    
    for pattern_name, details in patterns.items():
        print(f"\n{pattern_name}")
        print("-" * 40)
        print(f"Description: {details['Description']}\n")
        print(f"Example:{details['Example']}\n")
        print(f"Used in: {details['When Used']}")


def example_5_head_importance_analysis():
    """Analyze which heads are most important."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Head Importance Analysis")
    print("=" * 60)
    
    print("""
HOW TO MEASURE HEAD IMPORTANCE:

1. ATTENTION ENTROPY
   - Measure how focused attention is
   - High entropy: Distributed attention
   - Low entropy: Focused attention
   - Lower entropy = more important (focused)

2. GRADIENT-BASED IMPORTANCE
   - How much does the head's output affect loss?
   - Higher gradient = more important
   - Requires backpropagation

3. ATTENTION PATTERN ANALYSIS
   - Do heads learn distinct patterns?
   - Similar patterns = redundant
   - Different patterns = important

4. ABLATION ANALYSIS
   - Remove head, measure accuracy drop
   - Larger drop = more important
   - Most reliable but expensive


EXAMPLE HEAD IMPORTANCE SCORES:

Layer 0 (Early):
  Head 0: 0.85 ★★★★★  (Position pattern - very important)
  Head 1: 0.72 ★★★★   (Broad attention - moderately important)
  Head 2: 0.45 ★★     (Redundant - less important)
  Head 3: 0.38 ★      (Very redundant - least important)

Layer 6 (Late):
  Head 0: 0.92 ★★★★★  (Semantic - very important)
  Head 1: 0.88 ★★★★★  (Grammar - very important)
  Head 2: 0.65 ★★★    (Intermediate - moderately important)
  Head 3: 0.42 ★      (Redundant - less important)


INSIGHTS:

- Early layers: Positional and syntactic attention important
- Late layers: Semantic attention important
- Some heads always less important (candidates for pruning)
- Redundancy exists for robustness
    """)


def example_6_head_pruning():
    """Explain pruning low-importance heads."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Attention Head Pruning")
    print("=" * 60)
    
    print("""
ATTENTION HEAD PRUNING:

Remove low-importance heads to:
1. Reduce model size
2. Speed up inference
3. Reduce memory
4. Minimal accuracy loss

PROCESS:

Step 1: Measure head importance
├─ Calculate importance scores
└─ Rank all heads

Step 2: Prune low-scoring heads
├─ Remove bottom 20-30% heads
├─ Adjust model dimensions
└─ Retrain briefly if needed

Step 3: Evaluate
├─ Test on validation set
├─ Check accuracy drop
└─ Adjust if needed


EXAMPLE RESULTS (DistilBERT):

Original:
- 12 layers × 6 heads = 72 heads
- Model size: 268 MB
- Inference time: 100ms

After pruning (50% heads):
- 6 layers × 3 heads = 18 heads
- Model size: 134 MB (50% smaller)
- Inference time: 55ms (45% faster)
- Accuracy drop: 0.5-1.5% (often acceptable)


TRADE-OFFS:

Advantages:
+ Smaller model
+ Faster inference
+ Less memory
+ Easier to deploy

Disadvantages:
- Slight accuracy loss
- Requires retraining/fine-tuning
- Need importance measurement
- Different tasks have different optimal configs


WHEN TO PRUNE:

✓ Do prune when:
  - Deploying to edge devices
  - Latency is critical
  - Storage is limited
  - Acceptable accuracy loss exists

✗ Don't prune when:
  - Maximum accuracy needed
  - Inference speed not critical
  - Model already small
    """)


def example_7_visualize_heatmap():
    """Create attention heatmap visualization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Attention Heatmap Visualization")
    print("=" * 60)
    
    print("""
CREATING ATTENTION HEATMAPS:

Heatmap shows attention matrix as color gradient

Example heatmap for "The cat sat on the mat":

              [CLS] The cat sat on the mat [SEP]
    [CLS]     ■■■  □  □  □  □  □  □   □    □
    The       ■■   ■¤  □  □  □  □  □   □    □
    cat       ■    ¤■¤  ◇  □  □  □   □    □
    sat       ■    □  ◇ ■■■  □  □   □    □
    on        ■    □  □  ¤  ■■  □   □    □
    the       ■    □  □  □  ◇ ■■   □    □
    mat       ■    □  □  □  □  ◇ ■■   □    □
    [SEP]     ■    □  □  □  □  □  □   ■    □


Colors:
■ = High attention (dark/red)
¤¤ = Medium attention (orange/yellow)
◇ = Low attention (light/blue)
□ = No attention (white)


INTERPRETATION:

From [CLS]:
- [CLS] attends to all tokens (collected global info)

From "cat":
- "cat" attends heavily to "sat" and "the"
- Shows noun-verb and article-noun relationships

From "sat":
- "sat" attends to "cat" (subject)
- "sat" attends to "on" and "mat" (location)
- Shows predicate structure

Common patterns:
- Diagonal: Position attention
- Column focus: Global token attention
- Clusters: Related word groups


TOOLS FOR VISUALIZATION:

1. Matplotlib
   - Create custom heatmaps
   - Good for presentations

2. Plotly
   - Interactive heatmaps
   - Zoom, hover for details

3. Bertviz
   - Specialized for attention
   - Multiple visualization types

4. Attention-is-all-you-need
   - Original paper style
   - Professional appearance
    """)


def example_8_multi_layer_analysis():
    """Show how attention evolves across layers."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Multi-Layer Attention Analysis")
    print("=" * 60)
    
    print("""
HOW ATTENTION EVOLVES ACROSS LAYERS:

12-layer BERT model shows progression:

LAYER 0-2 (Early layers):
- Learn positional relationships
- Syntactic patterns
- Token boundaries
- Local dependencies

Example:
"The quick brown fox"
  The ↔ quick (adjacent tokens)
  quick ↔ brown (adjacent tokens)
  brown ↔ fox (adjacent tokens)


LAYER 3-6 (Middle layers):
- Learn syntactic structure
- Grammar relationships
- Phrase boundaries
- Some long-range dependencies

Example:
"The quick brown fox jumps"
  jumps ← fox (subject-verb)
  fox ← quick, brown (modifiers)


LAYER 7-11 (Late layers):
- Learn semantic relationships
- Task-specific patterns
- Long-range dependencies
- Abstract concepts

Example for sentiment:
"This movie is great but plot is confusing"
- Positive words cluster together
- Negative words cluster together
- Models overall sentiment


ARCHITECTURE PROGRESSION:

Layer | Focus         | Pattern Type | Range
------|---------------|--------------|--------
0-2   | Position      | Local       | 1-3 tokens
3-6   | Syntax        | Medium      | 5-10 tokens
7-11  | Semantics     | Long-range  | All tokens


IMPLICATIONS:

For Transfer Learning:
- Lower layers: Reusable for many tasks (keep them)
- Middle layers: Task-specific (might fine-tune)
- Upper layers: Highly task-specific (might replace)

For Distillation:
- Small model learns from large model's attention
- Especially effective in middle layers

For Pruning:
- Early layers: Remove redundant heads carefully
- Late layers: More room for pruning
- Layer-by-layer optimization possible
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ATTENTION VISUALIZATION EXAMPLES")
    print("=" * 60)
    
    # Example 1: Basics
    example_1_attention_basics()
    
    # Example 2: Multi-head
    example_2_multi_head_attention()
    
    # Example 3: Visualize
    example_3_visualize_attention()
    
    # Example 4: Patterns
    example_4_attention_patterns()
    
    # Example 5: Head importance
    example_5_head_importance_analysis()
    
    # Example 6: Pruning
    example_6_head_pruning()
    
    # Example 7: Heatmap
    example_7_visualize_heatmap()
    
    # Example 8: Multi-layer
    example_8_multi_layer_analysis()
    
    print("\n" + "=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print("""
KEY TAKEAWAYS:

1. Attention: Model learns what to focus on
2. Multi-head: Different heads learn different patterns
3. Self-attention: Enables parallel processing
4. Visualization: Understand model decisions
5. Head importance: Some heads matter more
6. Pruning: Remove redundant heads for efficiency
7. Layer progression: Syntax → Semantics

NEXT STEPS:
- Visualize your own models' attention
- Measure head importance
- Experiment with head pruning
- Use attention for interpretability
- Fine-tune specific heads for your task
    """)


if __name__ == "__main__":
    main()
