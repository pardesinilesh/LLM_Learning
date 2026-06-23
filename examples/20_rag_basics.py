#!/usr/bin/env python3
"""
Example 20: Retrieval-Augmented Generation (RAG)

Learn how to build systems that retrieve relevant documents and use them to enhance generation.
Concepts: Semantic search, retrieval, augmented generation, knowledge bases
"""

import sys
from pathlib import Path


def example_1_rag_concept():
    """Explain RAG concept and why it's useful."""
    print("=" * 60)
    print("EXAMPLE 1: What is RAG?")
    print("=" * 60)
    
    print("""
RETRIEVAL-AUGMENTED GENERATION (RAG):

RAG combines two steps:
1. RETRIEVAL: Find relevant documents from a knowledge base
2. GENERATION: Use those documents to generate answer

WHY USE RAG?

Without RAG:
- LLM generates answer from memory
- May hallucinate (make up facts)
- Knowledge cutoff (outdated information)
- No source citations

With RAG:
- Retrieves relevant documents first
- Grounds answer in actual sources
- Can use latest information
- Can cite sources
- Factually accurate


THE RAG PIPELINE:

User Question
    ↓
[RETRIEVAL STAGE]
    ↓
1. Convert question to embedding
2. Search knowledge base for similar documents
3. Return top K relevant documents
    ↓
[AUGMENTATION]
    ↓
4. Combine question + retrieved documents
5. Create augmented prompt
    ↓
[GENERATION STAGE]
    ↓
6. Pass augmented prompt to LLM
7. Generate answer grounded in retrieved docs
    ↓
Answer with Source References


EXAMPLE:

Question: "What is the capital of Australia?"

WITHOUT RAG:
LLM: "The capital of Australia is Sydney"
(Wrong! May hallucinate)

WITH RAG:
Retrieval: Finds document "Australian Capital: Canberra is the capital..."
Augmented Prompt: "Based on the document: '...Canberra is the capital...'
                   What is the capital of Australia?"
LLM: "According to retrieved documents, the capital of Australia is 
      Canberra (Source: Australian Government Info)"
(Correct and sourced!)


USE CASES:

1. Question Answering over Documents
   - Search Wikipedia for relevant articles
   - Generate answer based on articles

2. Customer Support
   - Search knowledge base for relevant docs
   - Generate personalized response

3. Document Analysis
   - Extract info from large document collections
   - Answer questions about multiple documents

4. Medical/Legal Systems
   - Retrieve relevant regulations/case law
   - Generate accurate advice

5. Research Assistants
   - Find relevant papers
   - Summarize or answer questions about them
    """)


def example_2_create_knowledge_base():
    """Create a simple knowledge base."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Creating a Knowledge Base")
    print("=" * 60)
    
    # Sample knowledge base documents
    documents = [
        {
            "id": "doc_1",
            "title": "Neural Networks Basics",
            "content": """Neural networks are inspired by biological neurons in the brain.
They consist of layers of interconnected nodes called neurons.
Each neuron has weights and biases that are adjusted during training.
Neural networks can learn complex patterns from data.
They are the foundation of deep learning."""
        },
        {
            "id": "doc_2",
            "title": "Transformers Architecture",
            "content": """Transformers are a type of neural network architecture introduced in 2017.
They use self-attention mechanisms to process sequences.
Unlike RNNs, transformers process all tokens in parallel.
This makes them much faster to train and more efficient.
BERT, GPT, and T5 are examples of transformer models."""
        },
        {
            "id": "doc_3",
            "title": "Attention Mechanism",
            "content": """Attention mechanisms allow models to focus on relevant parts of input.
Self-attention compares each token with all other tokens.
This helps the model understand relationships between words.
Multi-head attention uses multiple attention mechanisms in parallel.
Attention is crucial for understanding context in language."""
        },
        {
            "id": "doc_4",
            "title": "Transfer Learning",
            "content": """Transfer learning uses knowledge from one task to help another task.
Pre-trained models are trained on large datasets first.
Then fine-tuned on smaller, task-specific datasets.
This requires less data and time than training from scratch.
Transfer learning is fundamental to modern NLP."""
        },
        {
            "id": "doc_5",
            "title": "Embeddings and Vectors",
            "content": """Embeddings convert text into numerical vectors.
Words with similar meanings have similar embeddings.
Word2Vec, GloVe, and contextual embeddings are common methods.
Embeddings capture semantic relationships between words.
They are essential for semantic search and similarity."""
        }
    ]
    
    print(f"\nKnowledge Base: {len(documents)} documents\n")
    
    for doc in documents:
        print(f"ID: {doc['id']}")
        print(f"Title: {doc['title']}")
        print(f"Content preview: {doc['content'][:80]}...\n")
    
    return documents


def example_3_embedding_and_indexing(documents):
    """Create embeddings for documents."""
    print("=" * 60)
    print("EXAMPLE 3: Creating Embeddings (Indexing)")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        import numpy as np
    except ImportError:
        print("ERROR: Required libraries not installed.")
        return None, None
    
    print("\n1. Loading embedding model...")
    
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Note: Could not load model: {e}")
        return None, None
    
    print("   ✓ Model loaded\n")
    
    print("2. Creating embeddings for each document...\n")
    
    embeddings_dict = {}
    
    for doc in documents:
        # Use title + content for embedding
        text = doc['title'] + " " + doc['content']
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get embedding
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        embeddings_dict[doc['id']] = embedding
        
        print(f"   ✓ Embedded '{doc['title']}'")
        print(f"     Embedding shape: {embedding.shape}")
    
    print(f"\n✓ Created {len(embeddings_dict)} embeddings\n")
    
    return embeddings_dict, documents


def example_4_semantic_retrieval(embeddings_dict, documents):
    """Retrieve documents based on query similarity."""
    print("=" * 60)
    print("EXAMPLE 4: Semantic Retrieval")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        from sklearn.metrics.pairwise import cosine_similarity
        import torch
        import numpy as np
    except ImportError:
        print("ERROR: Required libraries not installed.")
        return
    
    if embeddings_dict is None:
        return
    
    print("\n1. Loading model for query embedding...\n")
    
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except:
        return
    
    # Sample queries
    queries = [
        "How do neural networks work?",
        "What is attention mechanism in transformers?",
        "How is transfer learning used in NLP?",
    ]
    
    print("2. Retrieving documents for queries:\n")
    
    for query in queries:
        print(f"Query: '{query}'")
        print("-" * 40)
        
        # Embed query
        inputs = tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        query_embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        
        # Compute similarities
        similarities = {}
        
        doc_ids = list(embeddings_dict.keys())
        doc_embeddings = np.array([embeddings_dict[doc_id] for doc_id in doc_ids])
        
        # Cosine similarity
        sims = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        for doc_id, similarity in zip(doc_ids, sims):
            similarities[doc_id] = similarity
        
        # Sort by similarity (descending)
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Show top 3 results
        print("Retrieved documents (top 3):\n")
        for rank, (doc_id, similarity) in enumerate(sorted_sims[:3], 1):
            doc = next(d for d in documents if d['id'] == doc_id)
            print(f"  {rank}. {doc['title']}")
            print(f"     Similarity: {similarity:.3f}")
            print(f"     Preview: {doc['content'][:60]}...\n")


def example_5_augmented_prompting(documents):
    """Create augmented prompts with retrieved documents."""
    print("=" * 60)
    print("EXAMPLE 5: Augmented Prompting")
    print("=" * 60)
    
    print("\nAugmented prompt adds context from retrieved documents.\n")
    
    # Example retrieval result
    retrieved_docs = [
        documents[0],  # Neural Networks Basics
        documents[1],  # Transformers Architecture
    ]
    
    question = "How do transformers use attention?"
    
    print("1. ORIGINAL PROMPT (without retrieval):")
    print("-" * 40)
    original_prompt = f"Question: {question}\nAnswer:"
    print(original_prompt)
    print()
    
    print("2. AUGMENTED PROMPT (with retrieved documents):")
    print("-" * 40)
    
    # Create augmented prompt
    context = "\n\n".join([
        f"Document {i+1}: {doc['title']}\n{doc['content']}"
        for i, doc in enumerate(retrieved_docs)
    ])
    
    augmented_prompt = f"""Based on the following documents, answer the question.

DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:"""
    
    print(augmented_prompt)
    print()
    
    print("3. BENEFITS OF AUGMENTATION:")
    print("-" * 40)
    print("""
    • Provides factual context
    • Reduces hallucination
    • Enables sourcing
    • Handles knowledge cutoff
    • More reliable answers
    """)


def example_6_full_rag_pipeline(documents):
    """End-to-end RAG pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Complete RAG Pipeline")
    print("=" * 60)
    
    try:
        from transformers import pipeline, AutoTokenizer, AutoModel
        import numpy as np
    except ImportError:
        print("ERROR: Required libraries not installed.")
        return
    
    print("\n1. Initialize pipeline components...\n")
    
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1
        )
    except Exception as e:
        print(f"Note: Could not load models: {e}")
        return
    
    print("   ✓ All components loaded\n")
    
    print("2. RAG WORKFLOW:\n")
    
    workflow = """
    Step 1: USER QUESTION
    └─ "What is a neural network?"
    
    Step 2: RETRIEVE
    ├─ Embed question
    ├─ Search knowledge base
    └─ Return top 2 relevant documents
       - Neural Networks Basics (score: 0.92)
       - Transformers Architecture (score: 0.78)
    
    Step 3: AUGMENT
    ├─ Combine question + documents
    └─ Create augmented prompt:
       "Based on: [doc1], [doc2]...
        What is a neural network?"
    
    Step 4: GENERATE
    ├─ Pass to LLM generator
    └─ Generate answer:
       "Neural networks are systems inspired by biological..."
    
    Step 5: RETURN WITH SOURCES
    └─ Answer + citations to source docs
    """
    
    print(workflow)
    
    print("\n3. PSEUDO-CODE IMPLEMENTATION:\n")
    
    print("""
def rag_query(question, knowledge_base):
    # Step 1: Embed question
    question_embedding = embed(question)
    
    # Step 2: Retrieve similar documents
    relevant_docs = retrieve_top_k(
        question_embedding,
        knowledge_base,
        k=3
    )
    
    # Step 3: Create augmented prompt
    context = format_documents(relevant_docs)
    augmented_prompt = create_prompt(question, context)
    
    # Step 4: Generate answer
    answer = generate(augmented_prompt)
    
    # Step 5: Return with sources
    return {
        'answer': answer,
        'sources': [doc.title for doc in relevant_docs]
    }
    """)


def example_7_advanced_rag_techniques():
    """Advanced RAG techniques and improvements."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Advanced RAG Techniques")
    print("=" * 60)
    
    print("""
IMPROVEMENTS TO BASIC RAG:

1. HYBRID SEARCH
   Combine semantic + keyword search
   - Semantic: Find similar documents
   - Keyword: Find exact term matches
   - Better coverage of relevant docs

2. RERANKING
   Two-stage retrieval:
   - Stage 1: Fast retrieval (many candidates)
   - Stage 2: Rerank with expensive model
   - Better precision without speed loss

3. DENSE PASSAGE RETRIEVAL (DPR)
   - Index passages (not full documents)
   - Smaller, more focused chunks
   - Better granularity

4. HIERARCHICAL RETRIEVAL
   - Retrieve chapters → sections → passages
   - More structured search
   - Better for large documents

5. QUERY EXPANSION
   - Rewrite question in multiple ways
   - Retrieve for each variant
   - More coverage

6. DOCUMENT PREPROCESSING
   - Chunk documents smartly
   - Remove irrelevant parts
   - Keep important context

7. CACHING
   - Cache embeddings
   - Reuse for multiple queries
   - Faster responses

8. FEEDBACK LOOPS
   - Track which sources help most
   - Fine-tune retrieval
   - Improve over time


PARAMETER TUNING:

1. Number of documents (k)
   - More docs: More context, slower
   - k=3: Fast, focused
   - k=10: Comprehensive

2. Similarity threshold
   - Higher: More selective
   - Lower: Include more candidates

3. Chunk size
   - Larger: More context, but less specific
   - Smaller: More specific, but fragmented

4. Overlap between chunks
   - Helps maintain context
   - ~20% overlap is typical
    """)


def example_8_practical_applications():
    """Real-world RAG applications."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Practical Applications")
    print("=" * 60)
    
    print("""
REAL-WORLD USE CASES:

1. CUSTOMER SUPPORT CHATBOT
   Knowledge Base: FAQ, support articles, policies
   User: "How do I reset my password?"
   → Retrieves reset instructions
   → Generates personalized response
   → Cites support article

2. MEDICAL DIAGNOSIS ASSISTANT
   Knowledge Base: Medical literature, guidelines, case studies
   Query: "Patient with symptoms X, Y, Z"
   → Retrieves relevant conditions
   → References medical papers
   → Suggests diagnostic approach

3. LEGAL DOCUMENT ASSISTANT
   Knowledge Base: Laws, regulations, precedents
   Query: "Employment contract question"
   → Retrieves relevant statutes
   → Cites case law
   → Explains legal implications

4. RESEARCH PAPER ANALYZER
   Knowledge Base: Published papers
   Query: "Papers on attention mechanisms"
   → Retrieves relevant papers
   → Summarizes findings
   → Lists citations

5. KNOWLEDGE BASE Q&A
   Knowledge Base: Internal documentation
   Employee: "How do I submit expense report?"
   → Retrieves procedure
   → Answers with current policy
   → Links to forms

6. NEWS AGGREGATION
   Knowledge Base: Articles from multiple sources
   Topic: "Climate change policy"
   → Retrieves related articles
   → Generates summary
   → Cites sources

7. PRODUCT RECOMMENDATIONS
   Knowledge Base: Product catalog, reviews, specs
   User: "Looking for laptop under $1000"
   → Retrieves matching products
   → Summarizes features
   → Explains why it matches criteria


IMPLEMENTATION CHECKLIST:

- Prepare knowledge base (documents, formatting)
- Choose embedding model (small, fast vs large, accurate)
- Index embeddings (vector database, Faiss, etc.)
- Set retrieval parameters (number of docs, similarity)
- Test retrieval quality
- Add generation component
- Evaluate answer quality
- Monitor for hallucinations
- Collect feedback
- Improve iteratively
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("RETRIEVAL-AUGMENTED GENERATION (RAG) EXAMPLES")
    print("=" * 60)
    
    # Example 1: Concepts
    example_1_rag_concept()
    
    # Example 2: Create knowledge base
    documents = example_2_create_knowledge_base()
    
    # Example 3: Create embeddings
    embeddings_dict, docs_ref = example_3_embedding_and_indexing(documents)
    
    # Example 4: Retrieval
    if embeddings_dict is not None:
        example_4_semantic_retrieval(embeddings_dict, documents)
    
    # Example 5: Augmented prompting
    example_5_augmented_prompting(documents)
    
    # Example 6: Complete pipeline
    example_6_full_rag_pipeline(documents)
    
    # Example 7: Advanced techniques
    example_7_advanced_rag_techniques()
    
    # Example 8: Applications
    example_8_practical_applications()
    
    print("\n" + "=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print("""
KEY TAKEAWAYS:

1. RAG = Retrieval + Augmentation + Generation
2. Retrieves documents first, grounds answers in sources
3. Reduces hallucinations and enables fact-checking
4. Embeddings enable semantic search
5. Augmented prompts provide context
6. Works for many domains: support, legal, medical, research

NEXT STEPS:
- Set up vector database (Faiss, Pinecone, Weaviate)
- Prepare your knowledge base
- Create embeddings for your documents
- Build retrieval system
- Integrate with LLM
- Evaluate quality + iterate
- Monitor hallucinations
    """)


if __name__ == "__main__":
    main()
