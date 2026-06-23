#!/usr/bin/env python3
"""
Example 2: Embeddings and Vector Representations

Learn how tokens become numerical vectors that capture meaning.
Concepts: Embeddings, semantic similarity, vector space
"""

import sys


def example_1_basic_embeddings():
    """Show how to get embeddings for tokens."""
    print("=" * 60)
    print("EXAMPLE 1: Getting Embeddings")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        print("ERROR: Required libraries not installed.")
        print("Run: pip install -r requirements.txt")
        return
    
    print("\n1. Loading model and tokenizer...")
    try:
        model_name = "distilbert-base-uncased"  # Small, fast model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Note: Could not download model: {e}")
        print("Make sure you have internet connection.")
        return
    
    # Get embeddings for words
    print("\n2. Getting embeddings for individual words:\n")
    
    words = ["king", "queen", "man", "woman", "apple", "fruit"]
    
    embeddings_dict = {}
    for word in words:
        # Tokenize
        inputs = tokenizer(word, return_tensors="pt")
        
        # Get embeddings (forward pass)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract embedding (use [CLS] token or average)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        embeddings_dict[word] = embedding
        
        print(f"Word: '{word}'")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  First 10 values: {embedding[:10].tolist()}")
        print(f"  Min: {embedding.min():.4f}, Max: {embedding.max():.4f}")
        print()
    
    # Calculate similarity
    print("3. Semantic Similarity (cosine similarity):\n")
    
    def cosine_similarity(vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        return torch.nn.functional.cosine_similarity(
            vec1.unsqueeze(0), 
            vec2.unsqueeze(0)
        ).item()
    
    # Compare embeddings
    similarity_pairs = [
        ("king", "queen"),
        ("king", "man"),
        ("apple", "fruit"),
        ("apple", "king"),
        ("man", "woman"),
    ]
    
    for word1, word2 in similarity_pairs:
        sim = cosine_similarity(embeddings_dict[word1], embeddings_dict[word2])
        print(f"  {word1:8} ↔ {word2:8}: {sim:.4f}")
    
    print("\nINSIGHT: Similar words have similarity close to 1.0")
    print("         Different words have lower similarity")


def example_2_sentence_embeddings():
    """Show embeddings for sentences."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Sentence Embeddings")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        return
    
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception:
        return
    
    print("\n1. Getting embeddings for sentences:\n")
    
    sentences = [
        "I love cats",
        "I adore dogs",
        "The sky is blue",
        "Python is a programming language",
        "Dogs are great pets"
    ]
    
    sentence_embeddings = {}
    for sent in sentences:
        # Tokenize
        inputs = tokenizer(sent, return_tensors="pt", padding=True)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Average pooling for sentence representation
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        sentence_embeddings[sent] = embeddings
        
        print(f"Sentence: '{sent}'")
        print(f"  Embedding shape: {embeddings.shape}")
        print()
    
    # Calculate sentence similarity
    print("2. Sentence Similarity:\n")
    
    def cosine_similarity(vec1, vec2):
        return torch.nn.functional.cosine_similarity(
            vec1.unsqueeze(0), 
            vec2.unsqueeze(0)
        ).item()
    
    test_pairs = [
        ("I love cats", "I adore dogs"),
        ("I love cats", "The sky is blue"),
        ("Python is a programming language", "Dogs are great pets"),
    ]
    
    for sent1, sent2 in test_pairs:
        sim = cosine_similarity(
            sentence_embeddings[sent1],
            sentence_embeddings[sent2]
        )
        print(f"'{sent1}'")
        print(f"  ↔ '{sent2}'")
        print(f"  Similarity: {sim:.4f}\n")


def example_3_embedding_properties():
    """Explore properties of embeddings."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Embedding Properties")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        import numpy as np
    except ImportError:
        return
    
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception:
        return
    
    print("\n1. Embedding Dimension:")
    
    # Check embedding dimension
    text = "hello"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state[0, 0]
    print(f"  Model embedding dimension: {embedding.shape[0]}")
    print(f"  (Each token is represented by {embedding.shape[0]} numbers)")
    
    # Distribution of values
    print(f"\n2. Distribution of embedding values:")
    print(f"  Mean: {embedding.mean():.4f}")
    print(f"  Std Dev: {embedding.std():.4f}")
    print(f"  Min: {embedding.min():.4f}")
    print(f"  Max: {embedding.max():.4f}")
    
    print(f"\n3. Embedding values are normalized:")
    print(f"  (Most values are small, close to 0)")
    print(f"  This is typical for deeply learned embeddings")
    
    # Vector magnitude
    magnitude = torch.norm(embedding)
    print(f"\n4. Vector Magnitude (L2 norm):")
    print(f"  ||embedding|| = {magnitude:.4f}")
    print(f"  (Tells us the overall 'strength' of the embedding)")


def example_4_embedding_arithmetic():
    """Show vector arithmetic with embeddings."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Embedding Arithmetic")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        return
    
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception:
        return
    
    print("\nEmbeddings are vectors, so we can do arithmetic!")
    print("\nGetting embeddings for key concepts...\n")
    
    words = ["king", "queen", "man", "woman", "prince", "princess"]
    embeddings = {}
    
    for word in words:
        inputs = tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings[word] = outputs.last_hidden_state[:, 0, :].squeeze()
    
    print("Classic analogy: king - man + woman ≈ queen")
    print("\nLet's check this mathematically:")
    print()
    
    # Compute: king - man + woman
    result = embeddings["king"] - embeddings["man"] + embeddings["woman"]
    
    # Find closest word to result
    def find_closest(target_embedding, embeddings_dict):
        """Find closest embedding in dictionary."""
        def cosine_similarity(vec1, vec2):
            return torch.nn.functional.cosine_similarity(
                vec1.unsqueeze(0), vec2.unsqueeze(0)
            ).item()
        
        similarities = {
            word: cosine_similarity(target_embedding, emb)
            for word, emb in embeddings_dict.items()
        }
        return max(similarities, key=similarities.get)
    
    closest = find_closest(result, embeddings)
    
    print(f"Arithmetic operation: king - man + woman")
    print(f"Closest word in vocabulary: {closest}")
    print(f"Expected: queen, Got: {closest}")
    
    if closest == "queen":
        print("✅ SUCCESS! Vector arithmetic works!")
    else:
        print("(This might not work perfectly with this small model,")
        print(" but the concept is demonstrated!)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LEARNING OBJECTIVE")
    print("=" * 60)
    print("""
In this example, you'll learn:
1. What embeddings are (numerical vectors)
2. How to extract embeddings from models
3. How to measure semantic similarity
4. How embeddings capture meaning and relationships
5. Vector arithmetic with embeddings

Read docs/02_fundamentals.md for theory!
    """)
    
    example_1_basic_embeddings()
    example_2_sentence_embeddings()
    example_3_embedding_properties()
    example_4_embedding_arithmetic()
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    print("""
✓ Embeddings are vectors of numbers (typically 768-3072 dimensions)
✓ Similar words have similar embeddings (high cosine similarity)
✓ Vector arithmetic reveals semantic relationships
✓ Embeddings capture meaning learned from vast text data
✓ The same word always produces the same embedding

Next: Run example 3 (simple inference)
    """)
