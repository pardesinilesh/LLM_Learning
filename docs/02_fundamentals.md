# 02. Fundamentals: Tokenization and Embeddings

## What is Tokenization?

**Tokenization** is the process of breaking text into smaller pieces called **tokens** that the LLM can process.

### Why Tokenize?

LLMs work with numbers, not raw text. We need to convert text into numerical representations:

```
Text: "Hello, World!"
         ↓ (tokenization)
Tokens: ["Hello", ",", "World", "!"]
         ↓ (token conversion)
Token IDs: [15496, 11, 2082, 0]
         ↓ (embedding)
Vectors: [[0.2, -0.5, 0.1, ...], [...], ...]
```

---

## Types of Tokenization

### 1. **Word-level Tokenization**
Split on spaces and punctuation:
```
"Hello, World!" → ["Hello", ",", "World", "!"]
```
❌ **Problem**: Vast vocabulary, many rare words

### 2. **Character-level Tokenization**
Split every character:
```
"Hello" → ["H", "e", "l", "l", "o"]
```
❌ **Problem**: Long sequences, harder to learn

### 3. **Subword Tokenization** ⭐ (Used by Modern LLMs)
Break into meaningful subunits:
```
"unimaginable" → ["un", "imagin", "able"]
"tokenization" → ["token", "ization"]
```
✅ **Benefits**:
- Handles rare words better
- Smaller vocabulary
- Works across languages

---

## Subword Tokenization: BPE

**Byte Pair Encoding (BPE)** is the most popular subword method.

### How BPE Works

**Step 1**: Start with individual characters
```
"chat" = ['c', 'h', 'a', 't']
```

**Step 2**: Merge most frequent pairs iteratively
```
Iteration 1: Merge 'c' + 'h' → 'ch'  (most common)
           "chat" = ['ch', 'a', 't']

Iteration 2: Merge 'ch' + 'a' → 'cha' (most common)
           "chat" = ['cha', 't']

Iteration 3: Stop at desired vocab size
```

### Result
A vocabulary of about 50,000 tokens that efficiently covers most languages!

---

## Why Tokenization Matters

### Example: How Token Count Affects Cost

```
Question: "What is machine learning?"

Word Tokens:        ["What", "is", "machine", "learning", "?"]     = 5 tokens
Character Tokens:   ["W", "a", "t", ..., "i", "n", "g", "?"]       = 27 tokens
Subword Tokens:     ["What", "is", "machine", "learning", "?"]     = 5 tokens
```

💡 **Fewer tokens = Lower cost and faster inference**

---

## What are Embeddings?

An **embedding** is a numerical representation of text that captures meaning.

### From Tokens to Embeddings

```
Token: "king"
Token ID: 2845
Embedding: [0.2, -0.5, 0.8, 0.1, -0.3, ..., 0.7]  (768 numbers)
                    ↑    ↑    ↑                      
           Each number captures some
           aspect of the word's meaning
```

### What Do Embeddings Capture?

Modern embeddings learn to encode:

1. **Semantic Similarity**
   ```
   Embedding("king") ≈ Embedding("queen")  (close in space)
   Embedding("king") ≠ Embedding("table")  (far apart)
   ```

2. **Relationships**
   ```
   king - man + woman ≈ queen
   Paris - France + Germany ≈ Berlin
   ```

3. **Context**
   The same word can have different embeddings depending on surrounding words

---

## Embedding Dimensions

LLMs use embeddings with specific sizes:

| Model | Embedding Dimension |
|-------|-------------------|
| GPT-2 | 768 |
| GPT-3 | 12,288 |
| BERT | 768 |
| LLaMA 7B | 4,096 |

### Why These Dimensions?

- **More dimensions** = More information, but slower and requires more memory
- **Fewer dimensions** = Faster, cheaper, but less expressiveness
- **Trade-off**: Find the sweet spot for your use case

---

## How LLMs Actually Work Now

With all pieces together:

```
1. Input:  "What is AI?"

2. Tokenize:  ["What", "is", "AI", "?"]

3. Embed:  [[0.2, -0.5, ...], [0.1, 0.3, ...], ..., [0.9, -0.2, ...]]
           (4 embeddings, 768 dimensions each)

4. Process:  Transformer blocks analyze relationships
            between tokens using Attention

5. Predict:  Output probability for next token
            "?" → Model predicts "It's" is most likely (85%)

6. Decode:  Convert token to word: "It's"

7. Repeat:  "What is AI? It's artificial..." (continue until done)
```

---

## Key Insight: Vector Space

Embeddings place words in **high-dimensional space** where:
- Similar words are close together
- Opposite words are far apart
- Relationships are geometric

### 2D Visualization (simplified from 768D)

```
      ↑ feminine
      |
female|  woman
      |     queen
      |________→ royalty
male  |    man
      |      king
      |
```

In reality, embeddings are in 768+ dimensions! We use dimension reduction techniques to visualize them.

---

## From Fundamentals to LLMs

Now you understand:

✅ **Tokenization**: Text → Numbers (tokens)
✅ **Embeddings**: Tokens → Vectors with meaning
✅ **Context**: How surrounding information changes representation

These are the building blocks for everything LLMs do!

---

## Next Concepts (Preview)

### Attention Mechanism
How does the model know which tokens are important to each other?

### Transformer Blocks
How does information flow through the model?

### Next Token Prediction
How does the model actually decode and generate text?

We'll explore these in `03_how_llms_work.md`

---

## Quick Reference

| Concept | What It Does | Why It Matters |
|---------|------------|---------------|
| **Token** | Smallest unit model processes | Determines input/output size |
| **Token ID** | Numeric ID for a token | How model accesses embeddings |
| **Embedding** | Vector representation | Captures semantic meaning |
| **Dimension** | Size of the vector | More = More info but slower |
| **Vocabulary** | Set of all possible tokens | Limits what model can express |

---

[← Back: Introduction](01_introduction.md) | [Next: How LLMs Work →](03_how_llms_work.md)
