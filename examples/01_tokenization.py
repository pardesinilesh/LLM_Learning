#!/usr/bin/env python3
"""
Example 1: Tokenization Basics

Learn how text is converted to tokens and back.
Concepts: Token IDs, vocabulary, BPE tokenization
"""

import sys

def example_1_basic_tokenization():
    """Show basic tokenization with transformers library."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Tokenization")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers library not installed.")
        print("Run: pip install -r requirements.txt")
        return
    
    # Load a small tokenizer (GPT-2 for simplicity)
    print("\n1. Loading GPT-2 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"Note: Could not download tokenizer: {e}")
        print("Make sure you have internet connection.")
        return
    
    # Example texts
    texts = [
        "Hello, World!",
        "LLMs are amazing.",
        "Tokenization breaks text into pieces.",
        "unimaginable"
    ]
    
    print("\n2. Tokenizing example texts:\n")
    
    for text in texts:
        # Tokenize
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        
        print(f"Text: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Number of tokens: {len(tokens)}")
        print()
    
    # Vocabulary size
    print(f"3. Vocabulary Size: {len(tokenizer)}")
    print(f"   (This tokenizer can represent {len(tokenizer)} different tokens)")
    
    # Show special tokens
    print(f"\n4. Special Tokens:")
    print(f"   Start of Text: {tokenizer.bos_token_id} ({tokenizer.bos_token})")
    print(f"   End of Text: {tokenizer.eos_token_id} ({tokenizer.eos_token})")
    print(f"   Padding: {tokenizer.pad_token_id} ({tokenizer.pad_token})")
    print(f"   Unknown: {tokenizer.unk_token_id} ({tokenizer.unk_token})")
    
    # Detokenization (tokens back to text)
    print(f"\n5. Detokenization (converting back to text):")
    sample_text = "Machine learning is powerful."
    token_ids = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(token_ids)
    print(f"  Original: '{sample_text}'")
    print(f"  Token IDs: {token_ids}")
    print(f"  Decoded: '{decoded}'")
    
    # Show subword tokenization benefit
    print(f"\n6. Subword Tokenization (handling rare words):")
    long_words = [
        "unimaginable",
        "internationalization",
        "unprecedented"
    ]
    
    for word in long_words:
        tokens = tokenizer.tokenize(word)
        print(f"  '{word}' → {tokens}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    print("""
✓ Text is split into tokens (words, subwords, punctuation)
✓ Each token has a unique ID number
✓ Subword tokenization handles rare and unknown words
✓ The same text always tokenizes to same token IDs
✓ Vocabulary size is fixed (~50K for most LLMs)
    """)


def example_2_token_statistics():
    """Analyze tokenization of longer texts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Token Statistics")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        return
    
    # Different types of texts
    texts = {
        "Short sentence": "Hello world.",
        "Question": "What is machine learning and how does it work?",
        "Code": "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Long paragraph": """Machine learning is a subset of artificial intelligence that enables 
                           systems to learn from data without being explicitly programmed. 
                           It uses algorithms to find patterns and make predictions."""
    }
    
    print("\nTokenization statistics:\n")
    for text_type, text in texts.items():
        tokens = tokenizer.encode(text)
        words = text.split()
        ratio = len(tokens) / len(words)
        
        print(f"{text_type}:")
        print(f"  Words: {len(words)}")
        print(f"  Tokens: {len(tokens)}")
        print(f"  Ratio: {ratio:.2f} tokens per word")
        print()
    
    print("KEY INSIGHT: Code and technical text create more tokens!")


def example_3_custom_text():
    """Let users tokenize their own text."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Text Tokenization")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        return
    
    print("\nEnter your own text to tokenize (or press Enter to skip):")
    user_text = input("> ").strip()
    
    if user_text:
        tokens = tokenizer.tokenize(user_text)
        token_ids = tokenizer.encode(user_text)
        
        print(f"\nYour text: '{user_text}'")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Total tokens: {len(tokens)}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LEARNING OBJECTIVE")
    print("=" * 60)
    print("""
In this example, you'll learn:
1. How text is converted to tokens
2. What token IDs mean
3. How vocabulary works
4. Why subword tokenization is useful
5. How to tokenize and detokenize text

Read docs/02_fundamentals.md for theory!
    """)
    
    example_1_basic_tokenization()
    example_2_token_statistics()
    
    # Uncomment to interact
    # example_3_custom_text()
    
    print("\n✅ Tokenization example complete!")
    print("\nNext: Run example 2 (embeddings)")
