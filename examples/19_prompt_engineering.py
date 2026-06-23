#!/usr/bin/env python3
"""
Example 19: Prompt Engineering

Learn how to craft effective prompts to get better outputs from LLMs.
Concepts: Prompt templates, few-shot learning, chain-of-thought, system prompts
"""

import sys


def example_1_basic_prompting():
    """Demonstrate basic prompt patterns."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Prompting Techniques")
    print("=" * 60)
    
    try:
        from transformers import pipeline
    except ImportError:
        print("ERROR: transformers library not installed.")
        print("Run: pip install -r requirements.txt")
        return
    
    print("\nLoading text generation model...\n")
    
    try:
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1  # CPU
        )
    except Exception as e:
        print(f"Note: Could not load model: {e}")
        return
    
    print("1. SIMPLE PROMPT (Direct question)")
    print("-" * 40)
    
    prompt1 = "What is machine learning?"
    print(f"Prompt: {prompt1}\n")
    
    result = generator(prompt1, max_length=80, do_sample=True, temperature=0.7)
    print(f"Output: {result[0]['generated_text']}\n")
    
    # Example 2
    print("2. DETAILED PROMPT (More context)")
    print("-" * 40)
    
    prompt2 = """Explain machine learning in simple terms for a beginner.
Include what it is, why it's useful, and one real-world example."""
    
    print(f"Prompt: {prompt2}\n")
    
    result = generator(prompt2, max_length=100, do_sample=True, temperature=0.7)
    print(f"Output: {result[0]['generated_text']}\n")
    
    # Example 3
    print("3. ROLE-BASED PROMPT (Ask model to take a role)")
    print("-" * 40)
    
    prompt3 = """You are a helpful AI tutor. 
Explain what tokenization means in the context of LLMs."""
    
    print(f"Prompt: {prompt3}\n")
    
    result = generator(prompt3, max_length=100, do_sample=True, temperature=0.7)
    print(f"Output: {result[0]['generated_text']}\n")


def example_2_few_shot_learning():
    """Show how providing examples improves outputs."""
    print("=" * 60)
    print("EXAMPLE 2: Few-Shot Learning (In-Context Learning)")
    print("=" * 60)
    
    try:
        from transformers import pipeline
    except ImportError:
        return
    
    print("\nFew-shot learning means showing examples to teach the model what you want.\n")
    
    try:
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1
        )
    except:
        return
    
    # Zero-shot (no examples)
    print("1. ZERO-SHOT (No examples)")
    print("-" * 40)
    
    prompt_zero = """Classify the sentiment of: "I hate this product"
Sentiment:"""
    
    print(f"Prompt: {prompt_zero}\n")
    result = generator(prompt_zero, max_length=30, do_sample=True)
    print(f"Output: {result[0]['generated_text']}\n")
    
    # Few-shot (with examples)
    print("2. FEW-SHOT (With examples)")
    print("-" * 40)
    
    prompt_few = """Classify sentiment as positive or negative.

Examples:
Text: "I love this product"
Sentiment: positive

Text: "This is terrible"
Sentiment: negative

Text: "I hate this product"
Sentiment:"""
    
    print(f"Prompt: {prompt_few}\n")
    result = generator(prompt_few, max_length=40, do_sample=True)
    print(f"Output: {result[0]['generated_text']}\n")


def example_3_chain_of_thought():
    """Show chain-of-thought prompting for complex reasoning."""
    print("=" * 60)
    print("EXAMPLE 3: Chain-of-Thought Prompting")
    print("=" * 60)
    
    try:
        from transformers import pipeline
    except ImportError:
        return
    
    print("""Chain-of-thought means asking the model to show its reasoning step by step.
This often leads to better answers for complex questions.\n""")
    
    try:
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1
        )
    except:
        return
    
    # Without chain-of-thought
    print("1. WITHOUT CHAIN-OF-THOUGHT")
    print("-" * 40)
    
    prompt_direct = """If a model has 100 million parameters and takes 2 bytes per parameter,
what is the model size in GB?
Answer:"""
    
    print(f"Prompt: {prompt_direct}\n")
    result = generator(prompt_direct, max_length=50, do_sample=True)
    print(f"Output: {result[0]['generated_text']}\n")
    
    # With chain-of-thought
    print("2. WITH CHAIN-OF-THOUGHT (Ask for steps)")
    print("-" * 40)
    
    prompt_cot = """If a model has 100 million parameters and takes 2 bytes per parameter,
what is the model size in GB?

Think step by step:
Step 1: Calculate total bytes
Step 2: Convert to GB
Answer:"""
    
    print(f"Prompt: {prompt_cot}\n")
    result = generator(prompt_cot, max_length=70, do_sample=True)
    print(f"Output: {result[0]['generated_text']}\n")


def example_4_prompt_templates():
    """Reusable prompt templates for common tasks."""
    print("=" * 60)
    print("EXAMPLE 4: Prompt Templates for Common Tasks")
    print("=" * 60)
    
    print("\nPrompt templates are reusable structures you fill in with your data.\n")
    
    # Template 1: Classification
    print("1. CLASSIFICATION TEMPLATE")
    print("-" * 40)
    
    classification_template = """Classify the following {input_type} as {categories}.

{input_type}: {user_input}

Classification: """
    
    example_input = {
        "input_type": "review",
        "categories": "positive, negative, or neutral",
        "user_input": "The product works well but arrived late."
    }
    
    prompt = classification_template.format(**example_input)
    print(f"Template:\n{classification_template}\n")
    print(f"Filled prompt:\n{prompt}\n")
    
    # Template 2: Summarization
    print("2. SUMMARIZATION TEMPLATE")
    print("-" * 40)
    
    summarization_template = """Summarize the following text in {length} sentences:

Text: {text}

Summary:"""
    
    example_input = {
        "length": "2-3",
        "text": "Machine learning is a type of artificial intelligence that enables systems to learn from data without being explicitly programmed. It's used in many applications like recommendation systems, image recognition, and natural language processing."
    }
    
    prompt = summarization_template.format(**example_input)
    print(f"Template:\n{summarization_template}\n")
    print(f"Filled prompt:\n{prompt}\n")
    
    # Template 3: Question Answering
    print("3. QUESTION ANSWERING TEMPLATE")
    print("-" * 40)
    
    qa_template = """Context: {context}

Question: {question}

Answer:"""
    
    example_input = {
        "context": "LLMs are large transformer-based models trained on massive amounts of text data. They can perform various NLP tasks like translation, summarization, and question answering.",
        "question": "What are LLMs trained on?"
    }
    
    prompt = qa_template.format(**example_input)
    print(f"Template:\n{qa_template}\n")
    print(f"Filled prompt:\n{prompt}\n")
    
    # Template 4: Creative Writing
    print("4. CREATIVE WRITING TEMPLATE")
    print("-" * 40)
    
    creative_template = """Write a {style} about {topic} in {length} words or less.

Constraints: {constraints}

Output:"""
    
    example_input = {
        "style": "short story",
        "topic": "time travel",
        "length": "100",
        "constraints": "Must include the words 'clock', 'tomorrow', and 'choice'"
    }
    
    prompt = creative_template.format(**example_input)
    print(f"Template:\n{creative_template}\n")
    print(f"Filled prompt:\n{prompt}\n")


def example_5_temperature_effects():
    """Show how temperature affects output diversity."""
    print("=" * 60)
    print("EXAMPLE 5: Temperature Effects on Output Diversity")
    print("=" * 60)
    
    try:
        from transformers import pipeline
    except ImportError:
        return
    
    print("""Temperature controls randomness in generation:
- 0.0 = Always pick most likely word (deterministic, boring)
- 0.5 = Balanced (some randomness, coherent)
- 1.0+ = Very random (creative, sometimes incoherent)\n""")
    
    try:
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1
        )
    except:
        return
    
    prompt = "The future of artificial intelligence will"
    
    temperatures = [0.3, 0.7, 1.2]
    
    for temp in temperatures:
        print(f"Temperature = {temp}")
        print("-" * 40)
        
        result = generator(
            prompt,
            max_length=50,
            do_sample=True,
            temperature=temp
        )
        
        generated_text = result[0]['generated_text'][len(prompt):]
        print(f"Generated: ...{generated_text}\n")


def example_6_system_prompts():
    """Explain system prompts for controlling model behavior."""
    print("=" * 60)
    print("EXAMPLE 6: System Prompts (Behavioral Control)")
    print("=" * 60)
    
    print("""System prompts (or system messages) control the model's personality and behavior.
They typically come before the user's actual question.\n""")
    
    # Example system prompts
    system_prompts = {
        "Expert Mode": """You are an expert in machine learning and artificial intelligence.
Provide detailed, technically accurate explanations.
Use precise terminology and cite examples when relevant.""",
        
        "Beginner Mode": """You are a friendly tutor explaining concepts to beginners.
Use simple language, avoid jargon, and use everyday analogies.
Always ask if the student has questions.""",
        
        "Code Helper": """You are an expert Python programmer.
Provide clear, well-commented code examples.
Explain the code and suggest best practices.""",
        
        "Skeptical Mode": """You are a critical thinker who questions claims.
Always consider limitations, caveats, and alternative perspectives.
Ask for evidence and reasoning.""",
    }
    
    print("EXAMPLE SYSTEM PROMPTS:\n")
    
    for mode, system_prompt in system_prompts.items():
        print(f"{mode}:")
        print("-" * 40)
        print(system_prompt)
        print()
    
    print("USAGE EXAMPLE:")
    print("-" * 40)
    print("""
    system_prompt = \"\"\"You are an expert Python programmer.
Provide clear, well-commented code.\"\"\"
    
    user_prompt = \"\"\"How do I sort a list of dictionaries by a specific key?\"\"\"
    
    # The model would then respond using both prompts combined
    """)


def example_7_prompt_best_practices():
    """Collection of best practices for effective prompting."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Prompt Engineering Best Practices")
    print("=" * 60)
    
    practices = {
        "1. Be Specific": {
            "❌ Bad": "Write about machine learning",
            "✓ Good": "Write a 3-paragraph explanation of how neural networks learn, suitable for someone with no ML background",
        },
        
        "2. Give Context": {
            "❌ Bad": "What is this?",
            "✓ Good": "Explain what a transformer architecture is and why it's important for LLMs. Assume the reader knows what neural networks are.",
        },
        
        "3. Use Examples": {
            "❌ Bad": "Classify this text",
            "✓ Good": """Classify the text as happy, sad, or neutral.

Examples:
- "I love this!" → happy
- "This is awful" → sad
- "The weather is cloudy" → neutral

Text to classify: "I'm okay with the decision" →""",
        },
        
        "4. Specify Output Format": {
            "❌ Bad": "List the benefits of Python",
            "✓ Good": "List 5 benefits of Python for data science. Format as a numbered list with one-sentence explanations.",
        },
        
        "5. Use Constraints": {
            "❌ Bad": "Summarize this article",
            "✓ Good": "Summarize this article in exactly 2-3 sentences using simple language.",
        },
        
        "6. Ask for Reasoning": {
            "❌ Bad": "Is this sentiment positive?",
            "✓ Good": "Is this sentiment positive? Explain your reasoning.",
        },
        
        "7. Iterate": {
            "Tip": "If the first response isn't great, refine the prompt and try again. Add more examples or be more specific.",
        },
    }
    
    print("\n")
    
    for practice, examples in practices.items():
        print(f"{practice}")
        print("-" * 40)
        
        for key, value in examples.items():
            if key == "Tip":
                print(f"{value}\n")
            else:
                print(f"{key}: {value}")
        print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PROMPT ENGINEERING EXAMPLES")
    print("=" * 60)
    
    # Example 1: Basic prompting
    example_1_basic_prompting()
    
    # Example 2: Few-shot learning
    example_2_few_shot_learning()
    
    # Example 3: Chain-of-thought
    example_3_chain_of_thought()
    
    # Example 4: Prompt templates
    example_4_prompt_templates()
    
    # Example 5: Temperature effects
    example_5_temperature_effects()
    
    # Example 6: System prompts
    example_6_system_prompts()
    
    # Example 7: Best practices
    example_7_prompt_best_practices()
    
    print("=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print("""
KEY TAKEAWAYS:

1. Clear, specific prompts → Better outputs
2. Examples (few-shot) help models understand patterns
3. Chain-of-thought improves reasoning
4. Templates make prompts reusable
5. Temperature controls randomness
6. System prompts set the model's behavior
7. Iteration helps refine prompts

NEXT STEPS:
- Experiment with your own prompts
- Try different templates for your tasks
- Combine techniques (system + few-shot + chain-of-thought)
- Measure quality and iterate
    """)


if __name__ == "__main__":
    main()
