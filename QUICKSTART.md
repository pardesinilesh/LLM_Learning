# Quick Start Guide

Get started with LLM learning in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 2GB disk space (for model downloads)
- Internet connection (first run only, to download models)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd LLM_LearningModels
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**First run:** This will download transformer models (~500MB total). This is normal and happens automatically.

## 5-Minute Quick Start

### Step 1: Read Introduction (2 min)

```bash
cat docs/01_introduction.md
```

This explains what LLMs are in simple terms.

### Step 2: Run Tokenization Example (1 min)

```bash
python examples/01_tokenization.py
```

See how text is converted to tokens!

### Step 3: Run Embeddings Example (1 min)

```bash
python examples/02_embeddings.py
```

Watch tokens become meaningful vectors.

### Step 4: Run Text Generation (1 min)

```bash
python examples/03_simple_llm.py
```

Generate text with a real LLM!

That's it! You've experienced the core LLM concepts. 🎉

---

## Learning Path

### For Beginners (2-3 hours)

1. Read: `docs/01_introduction.md` (15 min)
2. Run: `examples/01_tokenization.py` (10 min)
3. Read: `docs/02_fundamentals.md` (20 min)
4. Run: `examples/02_embeddings.py` (15 min)
5. Run: `examples/03_simple_llm.py` (15 min)
6. Experiment: Modify examples and see what happens (30 min)

### For Intermediate (4-6 hours)

1. Complete beginner path
2. Read: `docs/03_how_llms_work.md` (30 min)
3. Read: `docs/04_practical_guide.md` (30 min)
4. Run: `examples/04_fine_tuning_basics.py` (20 min)
5. Understand code in each example (60 min)
6. Modify examples for your use case (60 min)

### For Advanced (8+ hours)

1. Complete intermediate path
2. Implement: Create your own model or task
3. Experiment: Try different architectures and parameters
4. Deploy: Set up your own model serving

---

## Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'transformers'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Models downloading very slowly

**Solution:**
- Models are downloaded once and cached
- First run may take 5-10 minutes
- Subsequent runs are instant
- Check internet connection

### Issue: Out of memory error

**Solution:**
- Use small models (distilbert, distilgpt2)
- Examples already use small models
- For development, use CPU (slower but works)

### Issue: GPU not being used

**Solution:**
- Most examples run on CPU by default
- No GPU required for this tutorial!
- Optional: Install CUDA for faster inference

---

## Customization & Exploration

### Try Different Models

Edit example files to use different models:

```python
# Current:
model_name = "distilgpt2"

# Try these alternatives:
model_name = "gpt2"                          # Medium model
model_name = "facebook/opt-125m"             # Smaller but powerful
model_name = "google/t5-small"               # T5 architecture
```

### Experiment with Parameters

```python
# In 03_simple_llm.py, change temperature:
results = generator(
    prompt,
    temperature=0.2,  # Try: 0.1, 0.5, 1.0, 1.5, 2.0
)

# Change other parameters:
top_k=10,           # Try: 5, 20, 50
top_p=0.9,          # Try: 0.8, 0.95
max_length=50,      # Try: 20, 100, 200
```

### Create Your Own Examples

```python
# Follow the pattern of existing examples
# 1. Add clear docstring
# 2. Show step-by-step what's happening
# 3. Print intermediate results
# 4. Add learning insights at the end
```

---

## Next Steps

- 📖 **Read more**: Each code example has detailed comments
- 🔬 **Experiment**: Modify examples and observe changes
- 🚀 **Build**: Create your own LLM application
- 📚 **Learn deeper**: Check references in README.md

---

## Need Help?

1. Check the documentation in `docs/`
2. Read comments in code examples
3. Search error messages online
4. Try simpler examples first
5. Check model documentation on Hugging Face

---

## Tips for Success

✅ **Do:**
- Start small, understand one thing at a time
- Run examples multiple times
- Modify examples carefully
- Read error messages carefully
- Save working versions

❌ **Don't:**
- Try to understand everything at once
- Skip the theoretical parts
- Copy-paste without understanding
- Run on GPU your first time (CPU works fine!)
- Download the largest models

---

## Performance Notes

### Expected Runtime

- Tokenization: < 1 second
- Embeddings: 2-5 seconds (first run with download)
- Text generation: 5-30 seconds (depends on length)
- Fine-tuning: Not run by default (would take hours)

All examples are optimized for teaching, not performance!

### Memory Usage

- RAM needed: ~2GB
- GPU: Not required
- Storage: ~1-2GB for models (one-time download)

---

## What You'll Learn

After completing this course:

✅ What LLMs are and how they work
✅ Tokenization and embeddings
✅ Transformer architecture basics
✅ How models generate text
✅ Fine-tuning for your tasks
✅ Practical considerations
✅ Running inference

---

**Ready? Start with:** `python examples/01_tokenization.py`

Happy Learning! 🚀
