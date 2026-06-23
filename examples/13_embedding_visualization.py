#!/usr/bin/env python3
"""
Example 13: Embedding Visualization

Learn how to visualize embeddings in 2D/3D space to understand semantic relationships.
Concepts: t-SNE, dimensionality reduction, semantic similarity visualization
"""

import sys
from pathlib import Path


def example_1_extract_embeddings():
    """Extract embeddings from text samples."""
    print("=" * 60)
    print("EXAMPLE 1: Extracting Text Embeddings")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        import numpy as np
    except ImportError:
        print("ERROR: Required libraries not installed.")
        print("Run: pip install -r requirements.txt")
        return None, None
    
    print("\n1. Loading model and tokenizer...")
    
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Note: Could not load model: {e}")
        return None, None
    
    print("   ✓ Model loaded!\n")
    
    # Sample texts with semantic relationships
    texts = [
        # Animals
        "dog barking loudly",
        "cat sitting quietly",
        "bird flying high",
        "lion hunting prey",
        
        # Food
        "delicious pizza with cheese",
        "sweet chocolate cake",
        "fresh green salad",
        "spicy curry rice",
        
        # Technology
        "artificial intelligence systems",
        "machine learning algorithms",
        "neural networks deep learning",
        "computer vision models",
        
        # Emotions
        "happy and joyful moment",
        "sad and lonely feeling",
        "angry frustrated person",
        "calm peaceful mind",
    ]
    
    print("2. Sample texts for embedding:\n")
    for i, text in enumerate(texts, 1):
        print(f"   {i}. {text}")
    
    print(f"\n3. Extracting embeddings (dimension: 768)...\n")
    
    embeddings = []
    
    for i, text in enumerate(texts, 1):
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use mean pooling of all tokens
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        embeddings.append(embedding)
        
        print(f"   [{i:2d}/{len(texts)}] Embedded: '{text}'")
    
    embeddings = np.array(embeddings)
    print(f"\n   ✓ Extracted embeddings shape: {embeddings.shape}")
    print(f"     Each sentence → 768-dimensional vector\n")
    
    return embeddings, texts


def example_2_pca_visualization(embeddings, texts):
    """Visualize embeddings using PCA (simple, fast)."""
    print("=" * 60)
    print("EXAMPLE 2: PCA Dimensionality Reduction")
    print("=" * 60)
    
    try:
        from sklearn.decomposition import PCA
        import numpy as np
    except ImportError:
        print("ERROR: scikit-learn not installed.")
        print("Run: pip install scikit-learn")
        return None, None
    
    if embeddings is None:
        return None, None
    
    print("\n1. Applying PCA (Principal Component Analysis)...")
    print("   PCA finds the most important directions in the data\n")
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    print(f"   ✓ Reduced from 768D → 2D")
    print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"   Total variance explained: {pca.explained_variance_ratio_.sum():.2%}\n")
    
    print("2. PCA coordinates (first 5 samples):\n")
    for text, coords in zip(texts[:5], embeddings_2d[:5]):
        print(f"   '{text}'")
        print(f"     → PCA coordinates: ({coords[0]:.3f}, {coords[1]:.3f})\n")
    
    return embeddings_2d, pca


def example_3_tsne_visualization(embeddings, texts):
    """Visualize embeddings using t-SNE (better semantic structure)."""
    print("=" * 60)
    print("EXAMPLE 3: t-SNE Dimensionality Reduction")
    print("=" * 60)
    
    try:
        from sklearn.manifold import TSNE
        import numpy as np
    except ImportError:
        print("ERROR: scikit-learn not installed.")
        print("Run: pip install scikit-learn")
        return None
    
    if embeddings is None:
        return None
    
    print("\n1. Applying t-SNE (t-Distributed Stochastic Neighbor Embedding)...")
    print("   t-SNE preserves local semantic relationships better than PCA\n")
    print("   ⏳ This may take a moment...")
    
    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    print("   ✓ Completed!\n")
    
    print("t-SNE coordinates (first 5 samples):\n")
    for text, coords in zip(texts[:5], embeddings_2d[:5]):
        print(f"   '{text}'")
        print(f"     → t-SNE coordinates: ({coords[0]:.3f}, {coords[1]:.3f})\n")
    
    return embeddings_2d


def example_4_plot_visualization(pca_coords, tsne_coords, texts):
    """Create actual visualization plots."""
    print("=" * 60)
    print("EXAMPLE 4: Creating Visualization Plots")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib not installed.")
        print("Run: pip install matplotlib")
        return
    
    if pca_coords is None or tsne_coords is None:
        print("Cannot create plots without embeddings")
        return
    
    # Define categories for colors
    categories = {
        "Animals": ["dog barking loudly", "cat sitting quietly", "bird flying high", "lion hunting prey"],
        "Food": ["delicious pizza with cheese", "sweet chocolate cake", "fresh green salad", "spicy curry rice"],
        "Technology": ["artificial intelligence systems", "machine learning algorithms", "neural networks deep learning", "computer vision models"],
        "Emotions": ["happy and joyful moment", "sad and lonely feeling", "angry frustrated person", "calm peaceful mind"],
    }
    
    # Assign colors
    colors = {"Animals": "#FF6B6B", "Food": "#FFD93D", "Technology": "#6BCB77", "Emotions": "#4D96FF"}
    point_colors = []
    for text in texts:
        for category, texts_in_cat in categories.items():
            if text in texts_in_cat:
                point_colors.append(colors[category])
                break
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: PCA
    print("\n1. Creating PCA visualization...")
    for category, color in colors.items():
        mask = [point_colors[i] == color for i in range(len(point_colors))]
        ax1.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            label=category,
            color=color,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )
    
    ax1.set_xlabel("PCA Component 1", fontsize=11)
    ax1.set_ylabel("PCA Component 2", fontsize=11)
    ax1.set_title("PCA: Embedding Visualization", fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text labels
    for i, text in enumerate(texts):
        ax1.annotate(
            text[:15],  # Truncate for readability
            (pca_coords[i, 0], pca_coords[i, 1]),
            fontsize=7,
            alpha=0.6
        )
    
    print("   ✓ PCA plot created")
    
    # Plot 2: t-SNE
    print("   Creating t-SNE visualization...")
    for category, color in colors.items():
        mask = [point_colors[i] == color for i in range(len(point_colors))]
        ax2.scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            label=category,
            color=color,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )
    
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax2.set_title("t-SNE: Embedding Visualization", fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add text labels
    for i, text in enumerate(texts):
        ax2.annotate(
            text[:15],  # Truncate for readability
            (tsne_coords[i, 0], tsne_coords[i, 1]),
            fontsize=7,
            alpha=0.6
        )
    
    print("   ✓ t-SNE plot created\n")
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent / "embeddings_visualization.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"2. Plot saved to: {output_path}\n")
    
    # Show plot (if interactive)
    try:
        plt.show()
    except:
        print("   (Cannot display plot in non-interactive environment)")


def example_5_semantic_similarity_analysis(embeddings, texts):
    """Analyze semantic similarity using embeddings."""
    print("=" * 60)
    print("EXAMPLE 5: Semantic Similarity Analysis")
    print("=" * 60)
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        print("ERROR: scikit-learn not installed.")
        print("Run: pip install scikit-learn")
        return
    
    if embeddings is None:
        return
    
    print("\n1. Computing cosine similarity between all pairs...\n")
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    print("2. Finding most similar texts:\n")
    
    # For first few texts, find most similar
    for i in range(min(3, len(texts))):
        print(f"   Query: '{texts[i]}'")
        
        # Get similarities (excluding self)
        sims = similarity_matrix[i]
        # Find top 3 most similar (excluding self)
        top_indices = np.argsort(sims)[-4:-1]  # Top 3, excluding self
        
        for rank, j in enumerate(top_indices[::-1], 1):
            similarity = sims[j]
            print(f"      {rank}. '{texts[j]}' (similarity: {similarity:.3f})")
        print()
    
    print("3. Similarity insights:\n")
    print("   - Similar texts have similarity close to 1.0")
    print("   - Dissimilar texts have similarity close to 0.0")
    print("   - This helps find related content, detect duplicates, etc.")


def example_6_3d_visualization(embeddings, texts):
    """Create 3D visualization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: 3D Visualization with t-SNE")
    print("=" * 60)
    
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
    except ImportError:
        print("ERROR: Required libraries not installed.")
        print("Run: pip install scikit-learn matplotlib")
        return
    
    if embeddings is None:
        return
    
    print("\n1. Reducing embeddings to 3D...")
    print("   ⏳ This may take a moment...")
    
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=5)
    embeddings_3d = tsne_3d.fit_transform(embeddings)
    
    print("   ✓ Completed!\n")
    
    # Define categories for colors
    categories = {
        "Animals": ["dog barking loudly", "cat sitting quietly", "bird flying high", "lion hunting prey"],
        "Food": ["delicious pizza with cheese", "sweet chocolate cake", "fresh green salad", "spicy curry rice"],
        "Technology": ["artificial intelligence systems", "machine learning algorithms", "neural networks deep learning", "computer vision models"],
        "Emotions": ["happy and joyful moment", "sad and lonely feeling", "angry frustrated person", "calm peaceful mind"],
    }
    
    colors = {"Animals": "#FF6B6B", "Food": "#FFD93D", "Technology": "#6BCB77", "Emotions": "#4D96FF"}
    point_colors = []
    for text in texts:
        for category, texts_in_cat in categories.items():
            if text in texts_in_cat:
                point_colors.append(colors[category])
                break
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    print("2. Creating 3D plot...")
    
    for category, color in colors.items():
        mask = [point_colors[i] == color for i in range(len(point_colors))]
        ax.scatter(
            embeddings_3d[mask, 0],
            embeddings_3d[mask, 1],
            embeddings_3d[mask, 2],
            label=category,
            color=color,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.set_title("3D t-SNE Visualization of Embeddings", fontweight='bold')
    ax.legend()
    
    print("   ✓ 3D plot created\n")
    
    # Save
    output_path = Path(__file__).parent / "embeddings_3d_visualization.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"3. Plot saved to: {output_path}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("EMBEDDING VISUALIZATION EXAMPLES")
    print("=" * 60)
    
    # Example 1: Extract embeddings
    embeddings, texts = example_1_extract_embeddings()
    
    if embeddings is None:
        print("\nSkipping remaining examples due to embedding extraction error.")
        return
    
    # Example 2: PCA
    pca_coords, pca = example_2_pca_visualization(embeddings, texts)
    
    # Example 3: t-SNE
    tsne_coords = example_3_tsne_visualization(embeddings, texts)
    
    # Example 4: Plot visualizations
    if pca_coords is not None and tsne_coords is not None:
        example_4_plot_visualization(pca_coords, tsne_coords, texts)
    
    # Example 5: Similarity analysis
    example_5_semantic_similarity_analysis(embeddings, texts)
    
    # Example 6: 3D visualization
    example_6_3d_visualization(embeddings, texts)
    
    print("=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print("""
KEY TAKEAWAYS:

1. Embeddings capture semantic meaning in high-dimensional vectors
2. PCA: Fast dimensionality reduction, linear
3. t-SNE: Better at preserving local relationships, non-linear
4. Visualization helps understand what models learn
5. Similarity metrics (cosine) measure semantic closeness

NEXT STEPS:
- Compare different models' embeddings
- Use for semantic search and clustering
- Detect duplicate or similar documents
- Analyze specific domains (medical, legal, etc.)
    """)


if __name__ == "__main__":
    main()
