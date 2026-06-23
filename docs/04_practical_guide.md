# 04. Practical Guide: From Theory to Code

## Bridge Between Theory and Practice

Now that you understand **how LLMs work conceptually**, let's connect theory to actual code and practical applications.

---

## What Happens When You Interact with an LLM?

### You Type a Prompt

```
User: "Explain quantum computing in simple terms."
```

### Backend: Complete Process

**1. Tokenize**
```python
tokens = tokenizer.encode("Explain quantum computing in simple terms.")
# [37, 52, 4123, 15979, 16, 1405, 1439]
# (7 tokens)
```

**2. Create Embeddings**
```python
embeddings = model.embed(tokens)
# Shape: (7, 768)  - 7 tokens, 768 dimensions each
```

**3. Forward Pass Through 96 Transformer Layers**
```python
for layer in model.layers:
    embeddings = layer.attention(embeddings)      # Self-attention
    embeddings = layer.feed_forward(embeddings)   # FFN
    embeddings = layer.normalize(embeddings)      # Normalization
```

**4. Generate Predictions**
```python
# After all 96 layers

for i in range(max_tokens):  # Generate 1000+ tokens
    # Project to vocabulary space
    logits = model.output_projection(embeddings)  # (1, 50256)
    
    # Convert to probabilities
    probs = softmax(logits)  # (1, 50256)
    
    # Sample or take argmax
    next_token = sample(probs)
    
    # Append and continue
    tokens.append(next_token)
    embeddings = model.embed([next_token])  # Get embedding for new token
```

**5. Decode Tokens Back to Text**
```python
response = tokenizer.decode(tokens)
# "Quantum computing uses quantum bits or qubits..."
```

### Model Returns Response

```
"Quantum computing uses quantum bits (qubits) instead of regular bits.
A regular bit is either 0 or 1, but a qubit can be both at the same time,
thanks to something called superposition. This allows quantum computers
to explore many solutions simultaneously..."
```

---

## Key Practical Concepts

### 1. **Temperature** (Control Randomness)

```python
# Temperature = 0.0: Always pick most likely token (deterministic)
response = model.generate(prompt, temperature=0.0)
# Output: Always the same

# Temperature = 1.0: Standard probability distribution
response = model.generate(prompt, temperature=1.0)
# Output: Balanced randomness

# Temperature = 2.0: More random, more creative
response = model.generate(prompt, temperature=2.0)
# Output: Might be wild or nonsensical
```

**When to use:**
- **Creative tasks** (poetry, fiction): Higher temperature (0.7-1.2)
- **Factual tasks** (code, facts): Lower temperature (0.1-0.5)
- **Chatbot**: Medium (0.7)

### 2. **Top-K and Top-P Sampling**

```python
# Top-K: Only consider top K most likely tokens
response = model.generate(prompt, top_k=50)
# Eliminates tail of unlikely tokens

# Top-P (nucleus): Keep tokens that account for P% of probability
response = model.generate(prompt, top_p=0.9)
# Adaptive: keeps different number of tokens based on situation
```

### 3. **Max Tokens / Length Control**

```python
# Generate exactly 100 tokens maximum
response = model.generate(prompt, max_tokens=100)

# Keep generating until [END] token (model decides)
response = model.generate(prompt)
```

---

## Decoding Strategies

### Greedy Decoding (Simplest)

```
At each step: Pick the most likely next token

"The capital of France is"
  → "Paris" (99% likely)
  → "." (95% likely)
  → "France" (87% likely)

Result: "The capital of France is Paris.France..."
(Can get stuck in loops!)
```

### Beam Search (Better Quality)

```
Keep track of top N (e.g., 3) sequences:

Beam 1: "The capital of France is Paris"              (Score: 0.95 × 0.90)
Beam 2: "The capital of France is Paris, a city"    (Score: 0.95 × 0.50)
Beam 3: "The capital of France is Paris, France"    (Score: 0.95 × 0.45)

Pick the best scoring complete sequence
```

### Nucleus (Top-P) Sampling (Balanced)

```
At each step: Randomly sample from top P% of probable tokens

Probabilities: [0.40, 0.30, 0.15, 0.10, 0.05]
Top-90%:       [0.40, 0.30, 0.15]  ← Only sample from these
Remaining:     [0.10, 0.05]  ← Ignore

Random sample from [0.40, 0.30, 0.15]
```

**Benefits**: Prevents repetition, more natural, diverse

---

## Prompt Engineering Basics

### Technique 1: Be Specific

```
❌ Bad:   "Tell me about dogs"
✅ Good:  "Explain the major breeds of dog suitable for apartment living,
           including their temperament and exercise needs."
```

### Technique 2: Provide Context

```
❌ Bad:   "What is this?"
✅ Good:  "This is a code snippet. Explain what this Python function does:
           def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
```

### Technique 3: Chain of Thought

```
❌ Bad:   "What's 127 × 456?"
✅ Better: "Let's think through this step by step:
            What's 127 × 400?
            What's 127 × 50?
            What's 127 × 6?
            Add them up."
```

### Technique 4: Use Examples (Few-shot)

```
Classify the sentiment:

Example 1: "This movie is amazing!" → Positive
Example 2: "I hate waiting in lines." → Negative
Example 3: "The weather is okay today." → Neutral

Now classify: "This restaurant serves excellent food!" →
```

---

## Common Pitfalls and Fixes

### Pitfall 1: Hallucinations

```
❌ Problem:
   User: "Who won the 2089 World Cup?"
   Model: "Brazil won the 2089 World Cup defeating Argentina..."
   (2089 hasn't happened!)

✅ Fix: 
   - Add instructions: "Only use information from your training data."
   - Ask for reasoning: "What makes you confident in this answer?"
   - Adjust temperature lower for factual tasks
```

### Pitfall 2: Going Off-Topic

```
❌ Problem:
   User: "Write a poem about cats"
   Model: "Cats are great! Speaking of pets, did you know about hamsters..."

✅ Fix:
   - Be more explicit: "Write a poem about cats. Focus only on cats."
   - Shorter context window
   - Lower temperature
```

### Pitfall 3: Inconsistency

```
❌ Problem:
   Earlier: "I prefer dogs"
   Later: "I love cats more than any other animal"

✅ Fix:
   - Include context of previous statements
   - Ask explicitly: "Based on what you said earlier..."
   - Use system prompts to enforce consistency
```

---

## Working with Different Model Sizes

### Small Models (1-7B parameters)

**Pros**:
- Fast inference
- Low memory (can run on laptops)
- Cheap API costs
- Good for real-time applications

**Cons**:
- Less capable
- More hallucinations
- Weaker reasoning

**Best for**:
- Quick answers
- Classification
- Simple summarization
- Real-time applications

### Medium Models (13-40B parameters)

**Pros**:
- Good balance of speed and capability
- Reasonable memory requirements
- Better reasoning than small

**Cons**:
- Still hallucinate
- May need GPU with 24GB+ RAM

**Best for**:
- Content generation
- Code assistance
- Technical writing
- Fine-tuning projects

### Large Models (70B+ parameters)

**Pros**:
- Excellent reasoning
- Fewer hallucinations
- Handle complex tasks
- Better at novel reasoning

**Cons**:
- Slow inference
- Huge memory requirements
- Expensive
- Mostly accessed via API

**Best for**:
- Complex reasoning
- Novel problem-solving
- High-quality content
- Research

---

## The Model Inference Pipeline (Simplified)

```
User Input (Text)
    ↓
String Preprocessing
  (remove special chars, normalize)
    ↓
Tokenization
  (text → token IDs)
    ↓
Embedding Lookup
  (token IDs → vectors)
    ↓
Forward Pass (Transformer)
  (add positional encodings, run through layers)
    ↓
Logits (raw scores)
    ↓
Softmax (convert to probabilities)
    ↓
Sampling/Selection
  (temperature, top-k, top-p applied)
    ↓
Next Token (as ID)
    ↓
Detokenization
  (token → subword)
    ↓
Accumulate Output
    ↓
[Loop until [END] token or max_length]
    ↓
Final Output (Text)
```

---

## Hands-On Examples

You'll implement:

1. **Tokenization**: Understand different tokenizers
2. **Embeddings**: See how tokens become vectors
3. **Simple Inference**: Run a tiny model end-to-end
4. **Fine-tuning**: Adapt a model to your use case

Check the `examples/` folder!

---

## Quick Checklist Before Code

- [ ] Read the theory docs
- [ ] Understand tokenization
- [ ] Know what embeddings are
- [ ] Grasp transformer architecture
- [ ] Understand inference pipeline
- [ ] Have Python 3.8+ installed
- [ ] Install requirements.txt

---

[← Back: How LLMs Work](03_how_llms_work.md)
