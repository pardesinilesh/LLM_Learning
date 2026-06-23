#!/usr/bin/env python3
"""
Example 3: Simple LLM Inference

Learn how to run an actual LLM to generate text.
Concepts: Text generation, sampling, temperature, top-k sampling
"""

import sys


def example_1_basic_generation():
    """Generate text with an LLM."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Text Generation")
    print("=" * 60)
    
    try:
        from transformers import pipeline
    except ImportError:
        print("ERROR: transformers library not installed.")
        print("Run: pip install -r requirements.txt")
        return
    
    print("\n1. Loading text generation pipeline...")
    print("   (This loads a small, fast model)")
    
    try:
        # Use a small model that runs quickly
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1  # CPU (use 0 for GPU if available)
        )
    except Exception as e:
        print(f"Note: Could not load model: {e}")
        print("Make sure you have internet connection.")
        return
    
    print("   ✓ Model loaded!\n")
    
    # Generate text with different prompts
    prompts = [
        "Machine learning is",
        "The future of AI will",
        "Python is a programming language that"
    ]
    
    print("2. Generating text from prompts:\n")
    
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        
        # Generate
        results = generator(
            prompt,
            max_length=50,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
        
        generated_text = results[0]['generated_text']
        print(f"Output: '{generated_text}'\n")


def example_2_temperature_control():
    """Show effect of temperature on generation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Temperature Effects")
    print("=" * 60)
    
    try:
        from transformers import pipeline
    except ImportError:
        return
    
    try:
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1
        )
    except Exception:
        return
    
    prompt = "Artificial intelligence is"
    temperatures = [0.2, 0.7, 1.5]
    
    print(f"\nPrompt: '{prompt}'")
    print("\nGenerating with different temperatures:\n")
    
    for temp in temperatures:
        print(f"Temperature: {temp}")
        print("(Lower = more deterministic, Higher = more random)")
        
        results = generator(
            prompt,
            max_length=50,
            num_return_sequences=1,
            do_sample=True,
            temperature=temp
        )
        
        generated = results[0]['generated_text'][len(prompt):]
        print(f"Generated: {generated.strip()}\n")


def example_3_multiple_outputs():
    """Generate multiple outputs to see diversity."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multiple Outputs & Diversity")
    print("=" * 60)
    
    try:
        from transformers import pipeline
    except ImportError:
        return
    
    try:
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1
        )
    except Exception:
        return
    
    prompt = "One day I decided to"
    
    print(f"\nPrompt: '{prompt}'")
    print("\nGenerating 3 different completions:\n")
    
    results = generator(
        prompt,
        max_length=40,
        num_return_sequences=3,
        do_sample=True,
        temperature=1.0
    )
    
    for i, result in enumerate(results, 1):
        generated = result['generated_text'][len(prompt):]
        print(f"Output {i}: {generated.strip()}\n")
    
    print("Notice: Same prompt, different outputs!")
    print("(This is due to sampling at different temperatures)")


def example_4_sampling_strategies():
    """Compare different sampling strategies."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Sampling Strategies")
    print("=" * 60)
    
    try:
        from transformers import pipeline
    except ImportError:
        return
    
    try:
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1
        )
    except Exception:
        return
    
    prompt = "The capital of France is"
    
    print(f"\nPrompt: '{prompt}'")
    print("\nDifferent sampling strategies:\n")
    
    # Greedy (deterministic, always pick best)
    print("1. Greedy Decoding (deterministic):")
    results = generator(
        prompt,
        max_length=20,
        do_sample=False,  # No sampling = greedy
    )
    print(f"   Output: '{results[0]['generated_text'].strip()}'\n")
    
    # Random sampling
    print("2. Random Sampling (temperature=1.0):")
    results = generator(
        prompt,
        max_length=20,
        do_sample=True,
        temperature=1.0,
    )
    print(f"   Output: '{results[0]['generated_text'].strip()}'\n")
    
    # Top-K sampling
    print("3. Top-K Sampling (k=10):")
    results = generator(
        prompt,
        max_length=20,
        do_sample=True,
        top_k=10,
    )
    print(f"   Output: '{results[0]['generated_text'].strip()}'\n")
    
    # Top-P (nucleus) sampling
    print("4. Top-P / Nucleus Sampling (p=0.9):")
    results = generator(
        prompt,
        max_length=20,
        do_sample=True,
        top_p=0.9,
    )
    print(f"   Output: '{results[0]['generated_text'].strip()}'\n")


def example_5_generation_explained():
    """Show step-by-step what happens during generation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Generation Step-by-Step")
    print("=" * 60)
    
    print("""
Here's what happens when the model generates text:

1. INPUT PROMPT
   "The capital of France is"
   
2. TOKENIZATION
   ["The", "capital", "of", "France", "is"]
   → Token IDs: [464, 1218, 286, 3488, 318]
   
3. EMBEDDING LOOKUP
   Each token ID → 768D vector
   
4. FORWARD PASS (through layers)
   Process all tokens through all transformer layers
   Learn relationships: "France" ↔ "capital"
   
5. OUTPUT PROJECTION
   Last layer → probabilities for next token
   
6. SAMPLING
   Consider all 50,256 possible tokens
   Apply temperature/top-k/top-p
   Sample next most likely token
   
   Probabilities:
   - "Paris" : 0.95 (95%)    ← Selected!
   - "Lyon"  : 0.03 (3%)
   - "Nice"  : 0.01 (1%)
   - Other thousands: < 0.01%
   
7. APPEND TOKEN
   New prompt: "The capital of France is Paris"
   
8. LOOP
   Repeat steps 2-7 until:
   - [END] token generated, or
   - max_length reached, or
   - User stops
   
FINAL OUTPUT:
"The capital of France is Paris, located along the Seine River."
""")


def example_6_practical_tips():
    """Practical tips for generation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Practical Tips")
    print("=" * 60)
    
    print("""
PARAMETER TUNING GUIDE:

For Different Tasks:

1. FACTUAL / PRECISE OUTPUT
   - temperature: 0.1-0.3 (deterministic)
   - top_k: None
   - top_p: 0.9
   
   Example: Code generation, instructions
   
2. CREATIVE OUTPUT
   - temperature: 0.7-1.2 (balanced)
   - top_k: 50
   - top_p: 0.9
   
   Example: Creative writing, brainstorming
   
3. VERY CREATIVE OUTPUT
   - temperature: 1.5+ (random)
   - top_k: None
   - top_p: None
   
   Example: Poetry, unconventional ideas

4. BALANCED (DEFAULT)
   - temperature: 0.7
   - top_k: 50
   - top_p: 0.9
   
   Example: General conversation

COMMON MISTAKES TO AVOID:

❌ temperature too high
   → Model generates nonsense

❌ temperature too low
   → Model always says the same thing

❌ max_length too short
   → Incomplete answers

❌ max_length too long
   → Slow, repetitive generation

✅ BETTER: Start with defaults, adjust based on results
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LEARNING OBJECTIVE")
    print("=" * 60)
    print("""
In this example, you'll learn:
1. How to use a pre-trained LLM to generate text
2. The impact of temperature on output
3. Different sampling strategies
4. What happens step-by-step during generation
5. How to tune parameters for different tasks

Read docs/03_how_llms_work.md for deeper understanding!
    """)
    
    example_1_basic_generation()
    example_2_temperature_control()
    example_3_multiple_outputs()
    example_4_sampling_strategies()
    example_5_generation_explained()
    example_6_practical_tips()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
✓ Models process tokens through transformer layers
✓ Output is probability distribution over vocabulary
✓ Temperature controls randomness (0.1-2.0 typical)
✓ Sampling strategies affect output diversity
✓ Same prompt can generate different outputs

Next: Run example 4 (fine-tuning basics)
    """)
