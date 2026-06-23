# LLM Learning Models - Complete Index

## 🗺️ Navigation Guide

### Getting Started

**First Time?**
1. [QUICKSTART.md](QUICKSTART.md) - Install and run in 5 minutes
2. [README.md](README.md) - Understand the project
3. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Overview of what's included

### Learning Materials

**Theory Documents** (Read in order)

| # | Document | Topics | Time |
|---|----------|--------|------|
| 1 | [docs/01_introduction.md](docs/01_introduction.md) | What are LLMs, capabilities, limitations | 20 min |
| 2 | [docs/02_fundamentals.md](docs/02_fundamentals.md) | Tokenization, embeddings, vectors | 25 min |
| 3 | [docs/03_how_llms_work.md](docs/03_how_llms_work.md) | Transformers, attention, generation | 30 min |
| 4 | [docs/04_practical_guide.md](docs/04_practical_guide.md) | Development, optimization, best practices | 25 min |

**Total Theory Time:** ~2 hours of learning material

### Practical Examples

**Hands-On Tutorials** (Run in order)

| # | File | Concepts | Runtime |
|---|------|----------|---------|
| 1 | [examples/01_tokenization.py](examples/01_tokenization.py) | How text becomes tokens | 2 min |
| 2 | [examples/02_embeddings.py](examples/02_embeddings.py) | Tokens as vectors | 5 min |
| 3 | [examples/03_simple_llm.py](examples/03_simple_llm.py) | Text generation | 10 min |
| 4 | [examples/04_fine_tuning_basics.py](examples/04_fine_tuning_basics.py) | Model adaptation | 5 min |

**Total Code Time:** ~22 minutes

### Project Documentation

**Setup & Configuration**

| File | Purpose | Read When |
|------|---------|-----------|
| [QUICKSTART.md](QUICKSTART.md) | Fast setup guide | First time |
| [validate_setup.py](validate_setup.py) | Verify environment | After setup |
| [requirements.txt](requirements.txt) | Dependencies | Installing |
| [.gitignore](.gitignore) | Git configuration | Setting up repository |

**Contribution & Development**

| File | Purpose | Read When |
|------|---------|-----------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute | Before submitting PRs |
| [LICENSE](LICENSE) | MIT License | Using/sharing code |

**GitHub Actions**

| File | Purpose | Purpose |
|------|---------|---------|
| [.github/workflows/test.yml](.github/workflows/test.yml) | CI/CD automation | Testing on push |

### Data

| File | Purpose |
|------|---------|
| [data/sentiment_examples.csv](data/sentiment_examples.csv) | Sample sentiment dataset |

---

## 📊 Learning Roadmap

### Level 1: Beginner (3 hours)

**Goal:** Understand what LLMs are and see them in action

**Path:**
1. Read [01_introduction.md](docs/01_introduction.md)
2. Run [01_tokenization.py](examples/01_tokenization.py)
3. Read [02_fundamentals.md](docs/02_fundamentals.md)
4. Run [02_embeddings.py](examples/02_embeddings.py)
5. Run [03_simple_llm.py](examples/03_simple_llm.py)
6. Experiment: Modify temperatures, try different prompts

**Outcomes:**
- ✓ Understand tokenization
- ✓ Know what embeddings are
- ✓ Have used a real LLM
- ✓ Seen text generation in action

### Level 2: Intermediate (2-3 hours)

**Goal:** Understand how LLMs work under the hood

**Path:**
1. Read [03_how_llms_work.md](docs/03_how_llms_work.md)
2. Re-run previous examples
3. Read [04_practical_guide.md](docs/04_practical_guide.md)
4. Run [04_fine_tuning_basics.py](examples/04_fine_tuning_basics.py)
5. Deep dive: Read code comments in examples

**Outcomes:**
- ✓ Understand transformers
- ✓ Know attention mechanisms
- ✓ Understand fine-tuning
- ✓ Can optimize inference

### Level 3: Advanced (Variable)

**Goal:** Build your own LLM applications

**Path:**
1. Create custom examples
2. Fine-tune on your data
3. Deploy models
4. Read research papers
5. Contribute back

**Outcomes:**
- ✓ Can build LLM applications
- ✓ Understand performance tradeoffs
- ✓ Can fine-tune for specific tasks

---

## 🎯 By Use Case

### "I want to understand LLMs"

→ [01_introduction.md](docs/01_introduction.md) + 
→ [02_fundamentals.md](docs/02_fundamentals.md) +
→ Examples 1-2

### "I want to use LLMs"

→ [QUICKSTART.md](QUICKSTART.md) +
→ [03_simple_llm.py](examples/03_simple_llm.py) +
→ [04_practical_guide.md](docs/04_practical_guide.md)

### "I want to fine-tune models"

→ [03_how_llms_work.md](docs/03_how_llms_work.md) +
→ [04_fine_tuning_basics.py](examples/04_fine_tuning_basics.py) +
→ [04_practical_guide.md](docs/04_practical_guide.md)

### "I want to teach LLMs"

→ All theory docs +
→ All examples (as assignments) +
→ [CONTRIBUTING.md](CONTRIBUTING.md) (for extending)

### "I want to optimize inference"

→ [04_practical_guide.md](docs/04_practical_guide.md) +
→ Examples (for examples implementations)

---

## 📁 File Organization

```
Quick Reference:

Read First:
  README.md              ← Start here
  QUICKSTART.md          ← Setup help
  PROJECT_SUMMARY.md     ← Overview

Learn Theory:
  docs/01_introduction.md
  docs/02_fundamentals.md
  docs/03_how_llms_work.md
  docs/04_practical_guide.md

Try Code:
  examples/01_tokenization.py
  examples/02_embeddings.py
  examples/03_simple_llm.py
  examples/04_fine_tuning_basics.py

Setup:
  validate_setup.py      ← Check environment
  requirements.txt       ← Install dependencies
  .github/workflows/     ← CI/CD configuration

Contribute:
  CONTRIBUTING.md        ← How to contribute
  LICENSE                ← MIT License

Support:
  examples/utils.py      ← Shared utilities
  data/                  ← Sample data
```

---

## ⏱️ Time Estimates

| Activity | Time |
|----------|------|
| Quick start | 5 min |
| Read introduction & run first example | 30 min |
| Complete beginner path | 3 hours |
| Complete intermediate path | 2-3 hours |
| Advanced learning | Variable |

---

## 🔗 Cross References

**Theory explains examples:**
- 01_introduction.md → 01_tokenization.py
- 02_fundamentals.md → 02_embeddings.py
- 03_how_llms_work.md → 03_simple_llm.py (generation)
- 04_practical_guide.md → 04_fine_tuning_basics.py

**Examples reinforce theory:**
- Each example has reference to relevant theory doc
- Theory doc links to corresponding example
- All examples follow "theory → implementation" pattern

---

## 🎓 Recommended Sequence

**First Week:** Theory
- Day 1: Intro + foundations
- Day 2: How LLMs work
- Day 3: Practical considerations + deep dive

**Second Week:** Practice
- Day 4-5: Run and modify examples
- Day 6-7: Create your own implementation

**Ongoing:** Mastery
- Deeper papers and research
- Advanced fine-tuning
- Production deployment

---

## 📚 External References

Linked throughout the course:
- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Original Transformer Paper](https://arxiv.org/abs/1706.03762)
- [State of GPT Talk](https://www.youtube.com/watch?v=bZQun8Y4L2A)

---

## ❓ Quick FAQ

**"Where do I start?"**
→ [QUICKSTART.md](QUICKSTART.md)

**"How long will this take?"**
→ 3-4 hours for beginner path

**"Do I need GPU?"**
→ No, all examples run on CPU

**"What Python version?"**
→ 3.8+ (tested on 3.9, 3.10, 3.11)

**"Can I contribute?"**
→ Yes! See [CONTRIBUTING.md](CONTRIBUTING.md)

**"Can I use this for teaching?"**
→ Yes! MIT license allows it

**"Can I modify examples?"**
→ Yes! That's the point of learning

---

## ✅ Progress Checklist

- [ ] Read README.md
- [ ] Read QUICKSTART.md
- [ ] Run validate_setup.py
- [ ] Read docs/01_introduction.md
- [ ] Run examples/01_tokenization.py
- [ ] Read docs/02_fundamentals.md
- [ ] Run examples/02_embeddings.py
- [ ] Read docs/03_how_llms_work.md
- [ ] Run examples/03_simple_llm.py
- [ ] Read docs/04_practical_guide.md
- [ ] Run examples/04_fine_tuning_basics.py
- [ ] Modify an example (your choice)
- [ ] Create a custom example

---

**Table of Contents:**

- [QUICKSTART.md](QUICKSTART.md) ← Start here
- [README.md](README.md) ← Full overview
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) ← What's included
- [CONTRIBUTING.md](CONTRIBUTING.md) ← Contributing guide
- [LICENSE](LICENSE) ← MIT License

**Documentation:**
- [docs/01_introduction.md](docs/01_introduction.md)
- [docs/02_fundamentals.md](docs/02_fundamentals.md)
- [docs/03_how_llms_work.md](docs/03_how_llms_work.md)
- [docs/04_practical_guide.md](docs/04_practical_guide.md)

**Examples:**
- [examples/01_tokenization.py](examples/01_tokenization.py)
- [examples/02_embeddings.py](examples/02_embeddings.py)
- [examples/03_simple_llm.py](examples/03_simple_llm.py)
- [examples/04_fine_tuning_basics.py](examples/04_fine_tuning_basics.py)

---

**Ready to start?** → Open [README.md](README.md) 🚀
