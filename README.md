# LLM Learning Models - Beginner's Guide

A simple, practical introduction to Large Language Models (LLMs). This repository provides foundational knowledge, theory, and hands-on code examples to help you understand how LLMs work.

## 📚 What You'll Learn

- **Theory**: Core concepts behind Large Language Models
- **Fundamentals**: How tokenization, embeddings, and transformers work
- **Hands-On**: Practical code examples you can run immediately

## 🎯 Structure

```
├── docs/                    # Theory and concept explanations
│   ├── 01_introduction.md
│   ├── 02_fundamentals.md
│   ├── 03_how_llms_work.md
│   └── 04_practical_guide.md
├── examples/               # Runnable code examples
│   ├── 01_tokenization.py
│   ├── 02_embeddings.py
│   ├── 03_simple_llm.py
│   └── 04_fine_tuning_basics.py
├── tests/                 # Test examples
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo>
cd LLM_LearningModels

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Examples

```bash
# Run tokenization example
python examples/01_tokenization.py

# Run embeddings example
python examples/02_embeddings.py

# Run simple LLM inference
python examples/03_simple_llm.py

# Run fine-tuning basics
python examples/04_fine_tuning_basics.py
```

## 📖 Learning Path

1. **Start here**: Read [docs/01_introduction.md](docs/01_introduction.md) - Understand what LLMs are
2. **Learn fundamentals**: [docs/02_fundamentals.md](docs/02_fundamentals.md) - Core concepts
3. **Deep dive**: [docs/03_how_llms_work.md](docs/03_how_llms_work.md) - Architecture details
4. **Get hands-on**: Run examples in `examples/` folder in order
5. **Practice**: [docs/04_practical_guide.md](docs/04_practical_guide.md) - Tips and tricks

## 🔄 Automated Testing

All code examples are tested automatically using GitHub Actions:
- Tests run on every commit
- Validates code syntax and functionality
- Ensures examples are executable

You can see test results in the **Actions** tab of the repository.

## 📦 Dependencies

- `transformers` - Hugging Face library for LLMs
- `torch` - Deep learning framework
- `numpy` - Numerical computing
- `requests` - HTTP library
- `tqdm` - Progress bars

See `requirements.txt` for full list.

## 🎓 Key Topics Covered

- What are Large Language Models?
- Tokenization and vocabulary
- Word embeddings and representations
- Transformer architecture basics
- Attention mechanisms
- Fine-tuning and transfer learning
- Practical inference
- Prompting strategies

## 💡 Tips for Beginners

1. **Read before coding**: Always read the theory doc before running the example
2. **Run examples step-by-step**: Execute each example file to see how concepts work
3. **Modify and experiment**: Change parameters in examples to understand their impact
4. **Check outputs**: Most examples print intermediate results to aid understanding

## 🤝 Contributing

Found an issue or want to improve explanations? Feel free to contribute!

## 📝 License

This project is open source and available under the MIT License.

## 🔗 Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Attention is All You Need Paper](https://arxiv.org/abs/1706.03762)
- [State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A)

---

**Happy Learning!** 🚀

Questions or feedback? Open an issue in the repository.
