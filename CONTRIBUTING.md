# Contributing to LLM Learning Models

Thank you for your interest in contributing! We welcome improvements, new examples, and better explanations.

## Project Structure

```
├── docs/                      # Theory and concepts
│   ├── 01_introduction.md
│   ├── 02_fundamentals.md
│   ├── 03_how_llms_work.md
│   └── 04_practical_guide.md
├── examples/                  # Runnable code
│   ├── 01_tokenization.py
│   ├── 02_embeddings.py
│   ├── 03_simple_llm.py
│   └── 04_fine_tuning_basics.py
├── .github/workflows/         # CI/CD
│   └── test.yml
├── README.md                  # Main documentation
├── QUICKSTART.md              # Quick start guide
├── requirements.txt           # Dependencies
└── validate_setup.py          # Setup validation
```

## How to Contribute

### 1. Documentation Improvements

Clarify concepts or expand explanations:

**Good contributions:**
- Better analogies for complex concepts
- More examples
- Clearer explanations
- Links to additional resources
- Corrections to errors

**How to submit:**
1. Edit the appropriate doc in `docs/`
2. Keep explanations beginner-friendly
3. Add code examples where helpful
4. Test that links work
5. Create a pull request

### 2. New Examples

Add practical code examples:

**Good examples should:**
- Have a clear learning objective
- Start simple, build complexity
- Include detailed comments
- Print intermediate results (for learning)
- End with key insights
- Follow the existing format

**Template for new example (e.g., `05_tasks.py`):**

```python
#!/usr/bin/env python3
"""
Example N: Topic Name

Learn [what the example teaches].
Concepts: [key concepts covered]
"""

def example_1_basic():
    """Describe what this demonstrates."""
    print("=" * 60)
    print("EXAMPLE 1: Title")
    print("=" * 60)
    
    try:
        from transformers import ...
    except ImportError:
        print("ERROR: Required libraries not installed.")
        print("Run: pip install -r requirements.txt")
        return
    
    print("\n1. Clear step description...")
    # Implementation
    
    print("\n2. Next step...")
    # Implementation
    
    print("\nKEY INSIGHT: What you learned")


if __name__ == "__main__":
    print("\nLEARNING OBJECTIVE")
    print("What will learners understand?")
    
    example_1_basic()
    
    print("\nKEY TAKEAWAYS:")
    print("Summary of insights")
```

**How to submit:**
1. Create new file: `examples/0N_topic.py`
2. Follow the template above
3. Test that it runs without errors
4. Add to README.md table of contents
5. Create a pull request

### 3. Bug Fixes

Found an issue? Help us fix it!

**Before reporting:**
- Run `python validate_setup.py`
- Check existing issues
- Try with latest code

**When reporting:**
- Describe what happened
- Show error message
- Include Python version
- List installed package versions

### 4. Performance Improvements

Optimize examples:

**Can you improve:**
- Faster loading
- Lower memory usage
- Better code organization
- More efficient algorithms

**How to submit:**
- Keep functionality the same
- Add comments explaining optimizations
- Benchmark before/after if significant
- Create a pull request

## Development Guidelines

### Code Style

- Follow PEP 8
- Use clear variable names
- Add docstrings to functions
- Comment complex logic
- Keep functions focused

### Example Standards

Each example should:

1. **Start with clear output**
   ```python
   print("=" * 60)
   print("EXAMPLE N: Title")
   print("=" * 60)
   ```

2. **Show each step**
   ```python
   print("\n1. Step description...")
   # Implementation
   ```

3. **Print results**
   ```python
   print(f"Result: {result}")
   ```

4. **End with insights**
   ```python
   print("\nKEY INSIGHTS:")
   print("- What was learned")
   ```

### Documentation Standards

- Use Markdown formatting
- Include code blocks with syntax highlighting
- Add links internally with `[text](file.md)`
- Keep explanations concise but complete
- Include practical examples

### Testing

Before submitting:

```bash
# Validate setup
python validate_setup.py

# Run examples manually
python examples/01_tokenization.py
python examples/02_embeddings.py
python examples/03_simple_llm.py

# Check Python syntax
python -m py_compile examples/YOUR_FILE.py

# Test imports
python -c "from examples import ..."
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature
   ```bash
   git checkout -b feature/my-contribution
   ```
3. **Make changes** following guidelines
4. **Test thoroughly**
5. **Commit** with clear messages
   ```bash
   git commit -m "Add: [short description]"
   ```
6. **Push** to your fork
   ```bash
   git push origin feature/my-contribution
   ```
7. **Create Pull Request** with description
8. **Respond** to review feedback

## Commit Message Format

```
Type: Short description (50 chars)

Longer explanation if needed. Wrap at 72 characters.
Explain what and why, not how.

Fixes #123 (if applicable)
```

**Types:**
- `Add:` New feature or example
- `Fix:` Bug fix
- `Docs:` Documentation changes
- `Improve:` Performance or clarity
- `Refactor:` Code restructuring

## What Not to Contribute

❌ **Don't contribute:**
- Large model weights (commit to Git LFS instead)
- Copyrighted material
- Incomplete work (draft PRs OK with [WIP])
- Non-educational content
- Dependencies beyond scope (keep requirements minimal)

## Questions?

- Check existing issues/discussions
- Read the documentation
- Look at similar examples
- Ask in pull request comments

## Code of Conduct

- Be inclusive and respectful
- Focus on ideas, not individuals
- Help beginners learn
- Give constructive feedback
- Celebrate contributions

---

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in releases
- Credited in relevant files

Thank you for helping make LLM learning accessible! 🚀
