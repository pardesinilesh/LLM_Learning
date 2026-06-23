#!/usr/bin/env python3
"""
Example 8: Text Summarization

Learn how to summarize documents using abstractive and extractive methods.
Concepts: Summarization, abstractive vs extractive, multi-document summarization
"""

import sys
from pathlib import Path


def example_1_summarization_basics():
    """Explain summarization concepts."""
    print("=" * 60)
    print("EXAMPLE 1: Summarization Concepts")
    print("=" * 60)
    
    print("""
TWO APPROACHES TO SUMMARIZATION:

1. EXTRACTIVE SUMMARIZATION
   What it does: Copy important sentences from original text
   How: Score sentences, select top-scoring ones
   Pros: No new words, always grammatical
   Cons: May be choppy, doesn't rephrase ideas
   
   Example:
   Original: "The conference was held in Paris. 
              Over 2000 attended. Many new ideas were shared."
   Extracted summary: "The conference was held in Paris. 
                       Over 2000 attended."

2. ABSTRACTIVE SUMMARIZATION
   What it does: Generate new text capturing main ideas
   How: Neural network learns to paraphrase
   Pros: Fluent, can rephrase ideas, shorter
   Cons: May hallucinate, requires more computation
   
   Example:
   Original: "The conference was held in Paris. 
              Over 2000 attended. Many new ideas were shared."
   Abstractive summary: "Over 2000 people attended a Paris conference 
                         focused on sharing innovative ideas."

WHEN TO USE EACH:
- Extractive: News, legal documents (preserve exact wording)
- Abstractive: Blog posts, reports (coherent narratives)
    """)


def example_2_extractive_summarization():
    """Implement extractive summarization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Extractive Summarization")
    print("=" * 60)
    
    document = """
    Machine learning is a field of artificial intelligence that focuses on 
    building systems that can learn from data. Unlike traditional programming 
    where rules are explicitly coded, machine learning systems learn patterns 
    from examples. 
    
    There are three main types of machine learning: supervised learning, 
    unsupervised learning, and reinforcement learning. Supervised learning 
    uses labeled data to train models. Unsupervised learning finds patterns 
    in unlabeled data.
    
    Deep learning is a subset of machine learning using neural networks with 
    multiple layers. Deep learning has revolutionized computer vision and 
    natural language processing. Modern AI systems like ChatGPT use deep 
    learning transformers.
    
    The future of machine learning involves more efficient models, better 
    interpretability, and broader applications. Researchers are working on 
    models that require less data and computation.
    """
    
    print("\nOriginal document:")
    print("-" * 40)
    print(document.strip())
    print()
    
    # Split into sentences
    import re
    sentences = re.split(r'(?<=[.!?]) +', document.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    print(f"Total sentences: {len(sentences)}\n")
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}\n")
    
    print("=" * 40)
    print("Extractive Method 1: TF-IDF Scoring")
    print("=" * 40)
    
    print("""
TF-IDF (Term Frequency - Inverse Document Frequency):
- Measures importance of words in document
- Frequent words in this doc but rare overall = high score
- Sum TF-IDF scores for each sentence

Algorithm:
1. Split document into sentences
2. Calculate TF-IDF for each word
3. Score each sentence (sum of word scores)
4. Select top N sentences
5. Return in original order
    """)
    
    # Simple TF-IDF implementation
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
    except ImportError:
        print("ERROR: scikit-learn not installed.")
        return
    
    print("3. Scoring sentences with TF-IDF...\n")
    
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Score each sentence (sum of TF-IDF scores)
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    
    print("Sentence scores:")
    for i, (sent, score) in enumerate(zip(sentences, sentence_scores), 1):
        score_pct = score / sentence_scores.max() * 100
        bar = "█" * int(score_pct / 5)
        print(f"  {i}. [{bar:<20}] {score:.3f}")
    
    print("\n4. Selecting top 2 sentences for summary:\n")
    
    # Get top 2 sentences (keep original order)
    top_indices = np.argsort(sentence_scores)[-2:]
    top_indices = sorted(top_indices)  # Keep original order
    
    summary = ' '.join([sentences[i] for i in top_indices])
    
    print("Extractive Summary (TF-IDF):")
    print("-" * 40)
    print(summary)
    print()


def example_3_abstractive_summarization():
    """Use neural models for abstractive summarization."""
    print("=" * 60)
    print("EXAMPLE 3: Abstractive Summarization with Neural Networks")
    print("=" * 60)
    
    try:
        from transformers import pipeline
    except ImportError:
        print("ERROR: transformers library not installed.")
        return
    
    print("\n1. Loading summarization model...")
    print("   (Using BART: Denoising Autoencoder for sequence-to-sequence)")
    
    try:
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # CPU
        )
    except Exception as e:
        print(f"Note: Could not load model: {e}")
        print("Using smaller fallback model...")
        try:
            summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-6-6",
                device=-1
            )
        except:
            return
    
    print("   ✓ Model loaded!\n")
    
    # Sample documents
    documents = [
        {
            "title": "Climate Change",
            "text": """
Climate change poses one of the most significant challenges of our time. 
Global temperatures are rising due to increased greenhouse gas emissions. 
The primary cause is the burning of fossil fuels for energy. 
Effects include melting ice caps, rising sea levels, and extreme weather events.
Scientists recommend transitioning to renewable energy sources.
International cooperation through agreements like the Paris Climate Agreement
is essential to address this global crisis. 
Individual actions like reducing carbon footprint also matter.
            """.strip()
        },
        {
            "title": "Quantum Computing",
            "text": """
Quantum computing represents a revolutionary approach to computation.
Unlike classical computers using binary bits, quantum computers use quantum bits (qubits).
Qubits can exist in superposition, representing both 0 and 1 simultaneously.
This enables quantum computers to process vast amounts of data in parallel.
Quantum computers excel at specific problems like factoring large numbers,
drug discovery, and optimization problems.
However, they require extremely cold temperatures to operate.
Companies like IBM, Google, and others are developing practical quantum computers.
            """.strip()
        }
    ]
    
    print("2. Summarizing documents:\n")
    
    for doc in documents:
        print(f"Document: {doc['title']}")
        print("-" * 40)
        print(f"Original ({len(doc['text'].split())} words):")
        print(doc['text'][:100] + "...\n")
        
        # Summarize with different lengths
        for max_length in [50, 80, 120]:
            try:
                summary = summarizer(
                    doc['text'],
                    max_length=max_length,
                    min_length=min(30, max_length - 20),
                    do_sample=False
                )[0]['summary_text']
                
                word_count = len(summary.split())
                print(f"Summary (max {max_length} tokens, got {word_count} words):")
                print(f"  {summary}\n")
            except Exception as e:
                print(f"Could not generate summary: {e}\n")
    
    print()


def example_4_controlled_length_summarization():
    """Show how to control summary length."""
    print("=" * 60)
    print("EXAMPLE 4: Controlling Summary Length")
    print("=" * 60)
    
    print("""
Summarization parameters:

1. max_length
   - Maximum tokens in output summary
   - Limits verbosity
   - Example: max_length=100

2. min_length
   - Minimum tokens in output summary
   - Ensures sufficient detail
   - Example: min_length=50

3. do_sample
   - True: Use sampling (more creative, varied)
   - False: Use greedy decoding (deterministic)
   - Affects diversity and consistency

4. length_penalty
   - Positive: Prefer longer outputs
   - Negative: Prefer shorter outputs
   - Range: -2 to 2

5. num_beams (beam search)
   - Higher: Better quality, slower
   - Typical: 1 (greedy) to 4 (quality)

COMMON PATTERNS:

Short summary (1-2 sentences):
  max_length=50, min_length=20, do_sample=False

Medium summary (2-3 sentences):
  max_length=100, min_length=40, do_sample=False

Long summary (full details):
  max_length=200, min_length=80, do_sample=False

Creative summary:
  max_length=100, do_sample=True, temperature=0.7
    """)


def example_5_multi_document_summarization():
    """Summarize multiple documents together."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Multi-Document Summarization")
    print("=" * 60)
    
    print("""
Multi-document summarization combines information from multiple sources.

APPROACHES:

1. Concatenation
   - Join all documents
   - Summarize as single document
   - Simple but may be too long for model

2. Hierarchical
   - Summarize each document separately
   - Summarize the summaries
   - Handles long content

3. Query-focused
   - Summarize based on specific question
   - Extract relevant parts
   - More targeted results

EXAMPLE WORKFLOW:

1. Collect documents from multiple sources
2. Process separately or together
3. Extract key points
4. Remove redundancy
5. Produce coherent summary
    """)
    
    # Example: Multiple news articles
    articles = [
        {
            "source": "TechNews",
            "text": "AI company releases new language model with 70B parameters. Model outperforms previous versions on benchmarks."
        },
        {
            "source": "ScienceDaily",
            "text": "Researchers demonstrate advanced language model capabilities. The breakthrough enables new applications."
        },
        {
            "source": "AIWeekly",
            "text": "70 billion parameter model shows state-of-the-art performance. Trained on diverse data sources."
        }
    ]
    
    print("\nSample articles:\n")
    for article in articles:
        print(f"Source: {article['source']}")
        print(f"Content: {article['text']}\n")
    
    print("Combined text for summarization:")
    print("-" * 40)
    combined = " ".join([a['text'] for a in articles])
    print(combined)
    print("\n(In real application, this would be summarized by the model)")


def example_6_evaluation_metrics():
    """Explain summarization evaluation metrics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Evaluating Summarization Quality")
    print("=" * 60)
    
    print("""
AUTOMATIC METRICS:

1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
   - Most common metric for summarization
   - Measures overlap with reference summaries
   - Multiple variants:
     * ROUGE-1: Unigram overlap
     * ROUGE-2: Bigram overlap
     * ROUGE-L: Longest common subsequence

   Example calculation:
   Generated: "The cat sat on the mat"
   Reference: "The cat sat on the mat"
   ROUGE-1: 6/6 = 100% (all words match)
   
   Range: 0 to 1 (1 = perfect)
   Typical range: 0.3-0.5 (0.5 is very good)

2. BLEU (Bilingual Evaluation Understudy)
   - Originally for translation
   - Measures n-gram precision
   - Less ideal for summarization

3. METEOR
   - Considers synonyms and stems
   - More flexible than BLEU
   - Better for summary evaluation

HUMAN EVALUATION:

1. Informativeness
   - Does summary contain main ideas?
   - Score: 1-5

2. Coherence
   - Is summary well-written?
   - No grammar/spelling errors?
   - Score: 1-5

3. Conciseness
   - Is it brief yet complete?
   - Avoids redundancy?
   - Score: 1-5

BEST PRACTICE:
- Use automatic metrics for quick feedback
- Always validate with human evaluation
- Consider domain-specific criteria
    """)


def example_7_production_summaries():
    """Show practical summarization use cases."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Production Use Cases")
    print("=" * 60)
    
    print("""
REAL-WORLD APPLICATIONS:

1. NEWS AGGREGATION
   Input: Multiple news articles
   Output: Concise summary of key points
   Length: 2-3 sentences
   
2. LEGAL DOCUMENTS
   Input: Long contracts, terms
   Output: Bullet-point summary of key terms
   Length: Varies, 1-2 pages
   
3. RESEARCH PAPERS
   Input: Full academic paper
   Output: Abstract
   Length: 100-250 words
   
4. CUSTOMER REVIEWS
   Input: Multiple product reviews
   Output: Summary of common themes
   Length: 1-2 sentences
   
5. MEETING NOTES
   Input: Meeting transcript
   Output: Key decisions and action items
   Length: Bullet points
   
6. EMAIL SUMMARIZATION
   Input: Long email thread
   Output: Main points summary
   Length: 1-2 paragraphs
   
7. SOCIAL MEDIA
   Input: Trending topic discussion
   Output: What happened summary
   Length: Tweet-length (280 chars)

IMPLEMENTATION CONSIDERATIONS:

1. Speed
   - Real-time vs batch processing
   - Model size affects latency
   
2. Quality
   - Use domain-specific models
   - Fine-tune on your data
   
3. Cost
   - Smaller models are cheaper
   - Trade accuracy for speed
   
4. Consistency
   - Same input should give similar output
   - Use deterministic decoding
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TEXT SUMMARIZATION EXAMPLES")
    print("=" * 60)
    
    # Example 1: Concepts
    example_1_summarization_basics()
    
    # Example 2: Extractive
    example_2_extractive_summarization()
    
    # Example 3: Abstractive
    example_3_abstractive_summarization()
    
    # Example 4: Length control
    example_4_controlled_length_summarization()
    
    # Example 5: Multi-document
    example_5_multi_document_summarization()
    
    # Example 6: Evaluation
    example_6_evaluation_metrics()
    
    # Example 7: Production uses
    example_7_production_summaries()
    
    print("=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print("""
KEY TAKEAWAYS:

1. Two approaches: Extractive (copy sentences) vs Abstractive (rephrase)
2. Abstractive is more flexible but requires neural models
3. Control summary length with max_length, min_length
4. Evaluate with ROUGE metrics + human judgment
5. Different domains need different approaches
6. Multi-document summarization requires careful handling

NEXT STEPS:
- Try summarization on your own documents
- Compare different model sizes
- Fine-tune models on domain-specific data
- Measure ROUGE scores against ground truth
    """)


if __name__ == "__main__":
    main()
