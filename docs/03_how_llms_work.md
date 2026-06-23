# 03. How LLMs Work: Transformers and Attention

## The Transformer Architecture

Modern LLMs are built on **Transformers**, a neural network architecture introduced in 2017. All major models (GPT, Claude, LLaMA) use it.

### High-Level Overview

```
Text Input
    ↓
Tokenization + Embedding
    ↓
Transformer Blocks (many stacked layers)
    ├─ Self-Attention (see relationships)
    ├─ Feed-Forward Network (process)
    └─ Normalization (stabilization)
    ↓ (repeat × 12-96 blocks)
    ↓
Output: Probability distribution over vocabulary
    ↓
Select next token
    ↓
Text Output
```

---

## Self-Attention: The Key Innovation

**Attention** answers the question: "Which tokens matter most for understanding this token?"

### Simple Example

```
Input: "The bank is next to the river"
       
Processing token "bank":
  - Should I focus on "The"? (article, not important)
  - Should I focus on "river"? (YES! context suggests bank = riverbank)
  - Should I focus on "is"? (grammar, less important)
  - Should I focus on other "bank"s in text? (weighted by relevance)
```

### How Attention Works (Simplified)

**Step 1: Project embeddings into Query, Key, Value**
```
Each token embedding → Query (what am I looking for?)
                   → Key (what am I about?)
                   → Value (what information do I have?)
```

**Step 2: Calculate relevance (attention scores)**
```
For each token pair, compute similarity:
attention_score = Query·Key / √(key_dimension)

Example scores for "bank":
- bank vs "The"    : 0.1 (low relevance)
- bank vs "river"  : 0.8 (HIGH relevance)
- bank vs "next"   : 0.2 (low relevance)
```

**Step 3: Convert to weights**
```
softmax([0.1, 0.8, 0.2, ...]) = [0.05, 0.60, 0.15, ...]
                                  ↑                ↑
                          percentages (sum to 1)
```

**Step 4: Weighted combination**
```
Updated representation = 0.05×Value("The") + 0.60×Value("river") + 0.15×Value("next") + ...
                         Should heavily incorporate "river" context!
```

### Visualization

```
Attention Pattern for "bank":

        The bank is next to the river
The    [0.2  0.1  0.1  0.1  0.1  0.2  0.2]
bank   [0.1  0.6  0.1  0.1  0.1  0.1  0.9] ← Focus on "river"!
is     [0.1  0.1  0.5  0.1  0.1  0.1  0.1]
next   [0.1  0.2  0.1  0.6  0.2  0.1  0.1]
to     [0.1  0.1  0.1  0.2  0.5  0.1  0.1]
the    [0.2  0.1  0.1  0.1  0.1  0.5  0.2]
river  [0.1  0.9  0.1  0.1  0.1  0.1  0.6] ← Focus on "bank"!
```

---

## Multi-Head Attention

Instead of one attention pattern, process multiple patterns in parallel:

```
Input
  ├─ Attention Head 1: Focus on semantic relationships
  │  Learns: "bank" ↔ "river" (location context)
  │
  ├─ Attention Head 2: Focus on grammatical structure
  │  Learns: pronouns → their referents
  │
  ├─ Attention Head 3: Focus on abstract concepts
  │  Learns: business terms, financial context
  │
  └─ Attention Head 4: Focus on...
  
All heads output → Concatenate → Process together
```

Each head learns different patterns!

---

## A Transformer Block

One complete transformer layer:

```
Input Embedding
    ↓
[MULTI-HEAD ATTENTION]
  - Process all tokens in parallel
  - Each token attends to all other tokens
  - Learn what matters for understanding
    ↓
Layer Normalization (stabilize values)
    ↓
Residual Connection (add original input back)
    ↓
[FEED-FORWARD NETWORK]
  - Dense layer: expand dimensionality (4×)
  - Activation function: introduce nonlinearity
  - Dense layer: compress back to original dim
  - Learn additional patterns
    ↓
Layer Normalization
    ↓
Residual Connection
    ↓
Output Embedding (ready for next block)
```

### Why Residual Connections?

```
Without residual:    Output = Block(Input)
With residual:       Output = Block(Input) + Input
                              
Benefits:
- Prevents gradient vanishing in deep networks
- Preserves original information
- Easier optimization
```

---

## Stacking Blocks

Modern LLMs have **many transformer blocks stacked**:

| Model | Layers |
|-------|--------|
| GPT-2 | 12 |
| GPT-3 Large | 96 |
| LLaMA 7B | 32 |
| LLaMA 70B | 80 |

### Information Flow

```
Layer 1:  Learns basic patterns (word relationships, simple grammar)
Layer 2:  Learns intermediate concepts
...
Layer 8:  Learns complex semantics
...
Final:    Produces final representation
          → Linear projection → Distribution over 50K tokens
```

Each layer builds on understanding from previous layers!

---

## Positional Encoding

**Problem**: Attention is permutation-invariant. It doesn't naturally know token order!

```
"The dog bit the man" 
"The man bit the dog"

Without position info, both could look identical to attention (not good!)
```

**Solution**: Encode position into embeddings

```
"The"  → Embedding + Position Encoding for position 0
"dog"  → Embedding + Position Encoding for position 1
"bit"  → Embedding + Position Encoding for position 2
...

Different positions have different encodings, so attention knows order!
```

---

## Complete Generation Process

Now the full picture:

```
1. INPUT: "Hello, how are"

2. TOKENIZE: 
   ["Hello", ",", "how", "are"]

3. EMBED + POSITIONAL ENCODING:
   Token embeddings + position info

4. LAYER 1 FORWARD:
   - Self-attention: Learn which tokens relate to which
   - Feed-forward: Process each token
   - Output: Enhanced representations

5. LAYER 2-12 FORWARD:
   - Repeat: deepen understanding
   - Each layer refines the representations

6. FINAL OUTPUT PROJECTION:
   [0.01, 0.05, ..., 0.08]  ← Distribution over 50,256 tokens
   
7. ARGMAX or SAMPLE:
   Find token with highest probability
   Most likely: token 345 = "you"

8. DECODE:
   Token 345 → "you"

9. APPEND and LOOP:
   New input: "Hello, how are you"
   Repeat from step 2 for next token (add "?", "doing", etc.)

10. STOP:
    Model generates [END] token or user stops
```

---

## Key Insights

### Parallelization
- All tokens can be processed in parallel within one layer
- **Much faster** than sequential RNNs
- Can handle long sequences efficiently

### Context Window
```
Maximum tokens the model can process: context window
GPT-3: 4,096 tokens
GPT-4: 128,000 tokens
Claude: 200,000 tokens (very large!)
```

Longer context = Can reference more of conversation

### Why Is This Powerful?

1. **Flexible attention**: Model learns what matters (data-driven)
2. **Long-range dependencies**: Can relate distant tokens
3. **Parallel processing**: Much faster than sequence models
4. **Transfer learning**: Pre-trained patterns transfer well

---

## Comparison: Why Transformers Beat Previous Models

| Aspect | RNN/LSTM | Transformer |
|--------|----------|-------------|
| **Parallelization** | Sequential (slow) | Parallel (fast) |
| **Long dependencies** | Hard (gradient issues) | Easy (direct attention) |
| **Training speed** | Slow | Fast |
| **Interpretability** | Black box | Attention patterns visible |
| **Scaling** | Limited | Excellent |

---

## Next: Hands-On

You now understand:

✅ Transformer architecture
✅ Self-attention mechanism
✅ How information flows through layers
✅ How tokens get processed and transformed
✅ How next token is selected

Time to **see it in code**! Check `examples/` folder.

---

## Quick Reference

| Component | Purpose |
|-----------|---------|
| **Tokenization** | Text → Numbers |
| **Embedding** | Numbers → Vectors with meaning |
| **Positional Encoding** | Add position information |
| **Self-Attention** | Learn what tokens matter to each other |
| **Feed-Forward** | Process and refine representations |
| **Layer Norm** | Stabilize and normalize values |
| **Residual Connection** | Preserve information across layers |
| **Multi-head** | Learn different types of patterns |

---

[← Back: Fundamentals](02_fundamentals.md) | [Next: Practical Guide →](04_practical_guide.md)
