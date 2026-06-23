#!/usr/bin/env python3
"""
Example 26: Parameter-Efficient Fine-Tuning

Learn how to fine-tune large models with minimal trainable parameters.
Concepts: LoRA, Prefix Tuning, Adapter modules, Parameter efficiency
"""

import sys
from pathlib import Path


def example_1_finetuning_challenges():
    """Explain challenges with traditional fine-tuning."""
    print("=" * 60)
    print("EXAMPLE 1: Fine-Tuning Challenges with Large Models")
    print("=" * 60)
    
    print("""
TRADITIONAL FULL FINE-TUNING PROBLEM:

Large Language Models are HUGE:
- GPT-3: 175 billion parameters
- BERT-Large: 340 million parameters
- LLaMA-70B: 70 billion parameters

Fine-tuning all parameters requires:
1. Lots of GPU memory (VRAM)
2. Lots of training time
3. High computational cost
4. Full model storage per fine-tune

EXAMPLE: Fine-tuning GPT-3 (175B parameters)

Full Fine-Tuning:
  Memory needed: 175B × 4 bytes × 4 = 2.8 TB RAM
  Cost: $100,000+ in GPU hours
  Time: Days to weeks
  Storage: 700 GB per fine-tuned copy

Not practical for most applications!

SOLUTION: PARAMETER-EFFICIENT FINE-TUNING (PEFT)

Train only a small percentage of parameters:
- LoRA: 0.01% to 10% of parameters
- Adapters: 1% to 5% of parameters
- Prefix Tuning: 0.1% to 1% of parameters

Benefits:
✓ Fit on single GPU (12GB-40GB)
✓ Train in hours
✓ Cheap (costs ~$100)
✓ Easy to store (50MB-500MB)
✓ Keep base model frozen (immutable)
✓ Mix multiple fine-tunes on same base


COMPARISON TABLE:

Method              | Trainable | Memory | Training | Cost   | Quality
--------------------|-----------|--------|----------|--------|--------
Full Fine-Tuning   | 100%      | 2.8TB  | 7 days   | 100k   | ★★★★★
LoRA (rank 8)      | 0.1%      | 40GB   | 4 hours  | 100    | ★★★★☆
LoRA (rank 64)     | 1%        | 80GB   | 6 hours  | 150    | ★★★★★
Adapters           | 5%        | 100GB  | 5 hours  | 120    | ★★★★☆
Prefix Tuning      | 0.5%      | 50GB   | 3 hours  | 90     | ★★★☆☆

Key Insight: LoRA with rank 64 achieves near-full quality with 0.1% parameters!
    """)


def example_2_lora_explained():
    """Explain Low-Rank Adaptation (LoRA)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Low-Rank Adaptation (LoRA)")
    print("=" * 60)
    
    print("""
WHAT IS LORA?

Insight: Most weight changes needed for fine-tuning are low-rank
(can be expressed with few dimensions)

TRADITIONAL FINE-TUNING:
Original weight matrix W (e.g., 768 × 768 = 590,416 parameters)
- Update every single parameter
- Very expensive

LORA APPROACH:
Instead of updating W directly, add a low-rank decomposition:

W_new = W_original + ΔW

Where ΔW = B × A (low-rank factorization)
         (768×r) × (r×768) = 768×768

But we only train A and B!


VISUAL EXPLANATION:

Original matrix (768×768):
████████████████████
████████████████████
████████████████████
████████████████████

Low-Rank Update (rank-8):
    
Down-projection (768→8):        Up-projection (8→768):
████████                        ████████████████████
████████                        ████████████████████
████████                        ████████████████████
████████                        ████████████████████


MATHEMATICAL FORMULA:

h = W_original × x + (B × A) × x
    ↑                    ↑
  frozen          trainable (low-rank)

Where:
- W_original: Original weight (frozen)
- B: Up-projection matrix (trainable)
- A: Down-projection matrix (trainable)
- x: Input
- r: Rank (small number like 8, 16, 64)


PARAMETER COMPARISON:

Model: BERT-Large (340M parameters)
Apply LoRA to query and value projections only

LoRA without compression:
- Trainable parameters: 340M (100%)
- Not useful!

LoRA with rank-8:
- Hidden size: 768
- Number of heads: 12
- Rank: 8
- Per-head: (768×8) + (8×768) = 12,288 parameters
- Total heads: 144 (12 heads × 12 layers)
- Total: 144 × 12,288 = 1.77M parameters
- Percentage: 1.77M / 340M = 0.52%

LoRA with rank-64:
- Similar calculation: ~14.2M parameters
- Percentage: 14.2M / 340M = 4.2%


WHY DOES LORA WORK?

Empirical observation from research:
1. Fine-tuning updates have low intrinsic dimensionality
2. Even small rank (8-64) captures most important changes
3. Higher ranks give diminishing returns
4. Base model contains most knowledge


CHOOSING RANK (r):

r = 1-4: Very aggressive compression
- Smallest model size
- Some quality loss
- Fast training

r = 8-16: Balanced (most common)
- Good compression
- Minimal quality loss
- Good speed

r = 32-64: High quality
- Better quality
- More parameters
- Slower training

r > 64: Rare (almost full fine-tuning)
- Marginal improvements
- Use full fine-tuning instead


WHERE TO APPLY LORA:

Query projections:    Heavy training (always use)
Value projections:    Heavy training (always use)
Key projections:      Light training (optional)
Output projections:   Light training (optional)
Feed-forward layers:  Heavy training (optional, expensive)

Common: Apply to Q and V projections only
Result: Small, fast, effective
    """)


def example_3_prefix_tuning():
    """Explain Prefix Tuning approach."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Prefix Tuning")
    print("=" * 60)
    
    print("""
WHAT IS PREFIX TUNING?

Instead of modifying weights, prepend trainable tokens to input.

TRADITIONAL APPROACH:
Modify weight matrices → Expensive

PREFIX TUNING:
Add learnable prefix → Cheap

VISUALIZATION:

Input: "Translate English to French: Hello"

Traditional Fine-Tuning:
[Update all model weights during training]
↓
Output: "Bonjour"

Prefix Tuning:
[p1][p2][p3][p4][p5] + "Translate English to French: Hello"
    ↑ Trainable prefix (e.g., 20 tokens)
    ↑ Rest of model frozen
↓
Output: "Bonjour"


HOW IT WORKS:

1. Original model input:
   x = [token1, token2, ..., tokenn]

2. With prefix:
   x' = [p1, p2, ..., pk, token1, token2, ..., tokenn]
         ↑ Trainable prefix (k tokens, ~20)
         
3. All model parameters frozen
4. Only use prefix parameters in training
5. Model learns to generate useful prefix


PARAMETER COUNT:

Prefix length: 20
Hidden dimension: 768
Attention heads: 12
Layers: 12

Trainable parameters per layer: 20 × 768 = 15,360
Total (12 layers): 15,360 × 12 = 184,320

For 340M parameter model:
- Percentage: 184K / 340M = 0.05%
- Very small!


ADVANTAGES:

✓ Minimal parameters
✓ Model easily reusable (frozen)
✓ Multiple tasks on same model (different prefixes)
✓ Fast inference (cache prefix)
✓ Domain adaptation
✓ Lifelong learning

DISADVANTAGES:

✗ Limited to task adaptation
✗ Quality lower than LoRA (usually)
✗ Prefix must be consistent length
✗ Harder to debug


APPLICATIONS:

1. Task adaptation
   - Sentiment: "SENTIMENT: " prefix
   - Translation: "TRANSLATE EN-FR: " prefix
   - Summarization: "SUMMARIZE: " prefix

2. Style transfer
   - Formal: "FORMAL: " prefix
   - Casual: "CASUAL: " prefix
   - Technical: "TECHNICAL: " prefix

3. Few-shot learning
   - Examples in prefix
   - Model learns from prefix context


COMPARISON: PREFIX TUNING vs LORA

Prefix Tuning:
- Frozen model weights
- Learnable prefix
- ~0.05% parameters
- Good for task adaptation
- Quality ~90% of full fine-tuning

LoRA:
- Frozen base weights
- Learnable low-rank updates
- ~0.5-5% parameters
- Better quality
- Quality ~95% of full fine-tuning

LoRA is generally preferred (better quality/efficiency trade-off)
    """)


def example_4_adapters():
    """Explain Adapter modules approach."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Adapter Modules")
    print("=" * 60)
    
    print("""
WHAT ARE ADAPTERS?

Small neural networks inserted between layers.
Compress → Expand → Pass through.

VISUALIZATION:

Traditional layer:
Input → [Linear 768→768] → Output

With Adapter:
Input → [Linear 768→64] → [ReLU] → [Linear 64→768] → Output
        ↑ Down-project                ↑ Up-project
        ↑ Trainable                   ↑ Trainable


HOW ADAPTERS WORK:

1. Down-project: 768 → 64 (compress)
   - Reduce dimensionality
   - Reduce parameters

2. Non-linearity: ReLU activation
   - Add expressiveness
   - Enable non-linear transformations

3. Up-project: 64 → 768 (expand back)
   - Match original dimension
   - Can be applied layer-wise

4. Skip connection: Input + adapter output
   - Preserve original behavior if adapter learns identity
   - Stable training


PARAMETER COUNT:

Adapter size:
- Down projection: 768 × 64 = 49,152
- Up projection: 64 × 768 = 49,152
- Total per adapter: ~100K parameters

Full model (12 layers):
- 100K × 12 = 1.2M trainable parameters
- For 340M model: 0.35%

Variants:
- Half adapters: Only down-project (~35K params)
- Bottleneck adapters (parameterizable reduction)
- Layer-wise vs single adapter


ADVANTAGES:

✓ Modular (add/remove adapters)
✓ Multiple adapters for multiple tasks
✓ Combinable (stack adapters)
✓ Clear structure
✓ Works with any model
✓ Layer control (selective adapter placement)

DISADVANTAGES:

✗ Inference overhead (extra computations)
✗ Slightly slower than native model
✗ Requires careful bottleneck sizing
✗ More hyperparameters to tune


APPLICATIONS:

1. Multi-task learning
   Task 1 Adapter
   ├─ Sentiment analysis
   ├─ NER
   └─ Question answering

2. Domain adaptation
   Domain 1 Adapter → Medical domain
   Domain 2 Adapter → Legal domain
   Domain 3 Adapter → Technical domain

3. Language adaptation
   Language 1 Adapter → Spanish
   Language 2 Adapter → French
   Language 3 Adapter → Hindi

4. Progressive fine-tuning
   Adapter 1: Base task
   Adapter 2: Build on Adapter 1
   Adapter 3: Specialize further


STACKING ADAPTERS:

Multiple adapters for same task:
Input → Adapter1 → Adapter2 → Adapter3 → Output

Benefits:
- Combine different aspects
- Combine task + domain specialization
- Build complex adaptations


COMPARISON: ADAPTERS vs LORA vs PREFIX

Method        | Parameters | Speed     | Quality | Modularity
--------------|-----------|-----------|---------|------------
Full Fine-Tune| 100%      | Baseline  | 100%    | No
LoRA          | 0.5-5%    | +0%       | 95-98%  | Yes
Adapters      | 0.5-1%    | -5-10%    | 93-97%  | Very High
Prefix Tune   | 0.05%     | +0%       | 90-95%  | Limited

LoRA: Best overall (speed + quality + efficiency)
Adapters: Best modularity (multiple tasks)
Prefix: Best parameter efficiency
    """)


def example_5_comparative_analysis():
    """Compare all parameter-efficient methods."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Comparative Analysis of PEFT Methods")
    print("=" * 60)
    
    print("""
COMPREHENSIVE COMPARISON:

1. INFORMATION CAPACITY

Full Fine-Tuning: Can change any parameter
LoRA: Limited to low-rank manifold
Adapters: Limited by bottleneck size
Prefix: Limited to input perturbation

Ranking by capacity: Full > LoRA > Adapters > Prefix


2. SPEED (Training per step)

                    Relative Speed
Full Fine-Tuning:   1.0x (baseline)
LoRA (rank 64):     0.8-0.9x (faster)
Adapters:           0.7-0.8x (faster)
Prefix Tuning:      0.85-0.95x (similar)

LoRA/Prefix fastest because fewer computations


3. INFERENCE SPEED

                    Relative Speed
Full Fine-Tuning:   1.0x (baseline)
LoRA:               1.0x (merged into weights)
Adapters:           0.9-0.95x (extra computation)
Prefix:             0.95-1.0x (just longer sequence)

LoRA+Merge: Fastest (merge update into base weights post-training)


4. MODEL SIZE

Model: 340M base parameters

Full Fine-Tuning:
  One task: 340M
  10 tasks: 3400M (need 10 copies)

LoRA (rank 64):
  One task: 340M + 14M = 354M (+4%)
  10 tasks: 340M + (10 × 14M) = 480M (+41%)

Adapters (bottleneck 64):
  One task: 340M + 1.2M = 341M (+0.3%)
  10 tasks: 340M + (10 × 1.2M) = 352M (+3%)

Prefix (length 20):
  One task: 340M + 0.18M = 340M (+0.05%)
  10 tasks: 340M + (10 × 0.18M) = 341M (+0.1%)

Winner for multi-task: Prefix > Adapters > LoRA


5. QUALITY (on downstream tasks)

GLUE benchmark (average):
Full Fine-Tuning:    85.0
LoRA (rank 64):      84.3 (-0.7%)
LoRA (rank 8):       83.8 (-1.2%)
Adapters:            84.1 (-0.9%)
Prefix (length 20):  82.5 (-2.5%)

LoRA best quality trade-off


6. EASE OF USE

Full Fine-Tuning:    Straightforward (change all weights)
LoRA:                Easy (use LoRA libraries)
Adapters:            Moderate (need adapter modules)
Prefix:              Easy (just add tokens)

Score: Prefix = LoRA > Adapters > Full


RECOMMENDATION MATRIX:

Scenario                           | Best Choice
------------------------------------|-------------------
Maximum quality needed             | Full Fine-Tuning
Balance quality & efficiency       | LoRA (rank 32-64)
Multiple tasks, limited storage    | Adapters
Minimal parameters                 | Prefix Tuning
Edge device deployment             | LoRA (rank 8-16)
Fast iteration                     | Prefix Tuning
Production multi-task system       | Adapters
Research/exploration               | LoRA
    """)


def example_6_lora_implementation():
    """Show LoRA implementation conceptually."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: LoRA Implementation Details")
    print("=" * 60)
    
    print("""
PRACTICAL LORA IMPLEMENTATION:

Step 1: Apply LoRA to Model

from peft import get_peft_model, LoraConfig

config = LoraConfig(
    r=64,                            # Rank
    lora_alpha=16,                   # Scaling factor
    target_modules=["query", "value"],  # Apply to Q,V
    lora_dropout=0.05,               # Regularization
    bias="none",                      # Don't fine-tune bias
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

Output:
trainable params: 14,250,000
all params: 354,250,000
trainable%: 4.02%


Step 2: Training

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


Step 3: Save Only LoRA Weights

# Saves only ~15MB instead of whole 340MB model
model.save_pretrained("lora_model")


Step 4: Load & Use

from peft import AutoPeftModelForSeq2SeqLM

model = AutoPeftModelForSeq2SeqLM.from_pretrained(
    "lora_model",
    device_map="auto"
)

# Or merge with base model
merged_model = model.merge_and_unload()


LoRA HYPERPARAMETERS:

r (Rank):
- 8: Ultra-light, -5% quality
- 16: Light, -3% quality
- 32: Balanced, -1% quality
- 64: Good, ~0% quality
- 128+: Rarely needed

lora_alpha:
- Controls effective learning rate
- Typical: alpha = 2 × r
- Higher: More aggressive learning

target_modules:
- Which layers to apply LoRA
- Common: ["q_proj", "v_proj"]
- Can include more for higher quality

lora_dropout:
- Regularization (prevent overfitting)
- Typical: 0.05-0.1

bias:
- "none": Don't touch bias
- "all": Fine-tune bias too
- Usually "none" is fine


MERGING STRATEGY:

1. During Training:
   Keep LoRA separate, base frozen

2. For Inference:
   Option A: Load both base + LoRA
   Option B: Merge weights:
   W_merged = W_base + (alpha/r) × (B × A)
   Delete LoRA, use only W_merged
   
3. Multi-Task Inference:
   Load base once, load different LoRA per task
   Switch task → Load different LoRA
    """)


def example_7_training_considerations():
    """Training tips for parameter-efficient methods."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Training Considerations & Tips")
    print("=" * 60)
    
    print("""
TRAINING PARAMETER-EFFICIENT MODELS:

1. LEARNING RATE

LoRA:
  - Typically higher than full fine-tuning
  - Reason: Only updating small matrices
  - Recommended: 1e-3 to 1e-4
  - Check: 5e-4 is common starting point

Adapters:
  - Moderate: 2e-4 to 5e-4
  - Bottleneck size affects LR

Prefix Tuning:
  - Lower: 5e-5 to 1e-4
  - Very sensitive to hyperparameters

Full Fine-Tuning baseline:
  - 5e-5 for large models
  - But PEFT methods often use 10x higher


2. BATCH SIZE

LoRA: Can use larger batch size
  - Fewer parameters = less memory
  - Try 64-128 (vs 8-16 for full)

Adapters: Moderate batch size
  - 32-64 (extra computation cost)

Prefix: Standard batch size
  - 16-32 (similar to standard training)


3. WARMUP STEPS

LoRA: 500-1000 steps
- Shorter: Few parameters to stabilize

Full Fine-Tuning: 1000-2000 steps
- Longer: Many parameters need stabilization

Adapters: 500-1000 steps
- Similar to LoRA


4. EARLY STOPPING

PEFT methods can overfit faster:
- Monitor validation loss closely
- Might stop earlier than full fine-tuning
- Typical: 3-5 epochs vs 10+ for full

Check validation every N steps:
val_per_steps = 100


5. WEIGHT DECAY

LoRA: Moderate (0.01-0.1)
- Regularize low-rank updates

Full Fine-Tuning: Light (0.01)
- Huge parameter space

Prefix: None (0.0)
- Small parameter space, unnecessary


6. MIXED PRECISION

LoRA: FP16 safe
- Smaller matrices more stable
- Use with care

Full Fine-Tuning: FP16 tricky
- Large parameter space unstable
- Needs careful tuning

Recommended: FP32 for PEFT, FP16 if stable


7. GRADIENT ACCUMULATION

LoRA: Limited benefit
- Can fit larger batch anyway

Full Fine-Tuning: Essential
- Simulate larger batch
- Keep gradient = large_batch / accumulation_steps


SAMPLE TRAINING SETUP (LoRA):

model_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

training_config = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_steps=500,
    eval_steps=100,
    save_steps=500,
    logging_steps=50,
    fp16=True,  # Mixed precision
)

trainer = Trainer(
    model=peft_model,
    args=training_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
    """)


def example_8_production_deployment():
    """Deploy parameter-efficient models in production."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Production Deployment")
    print("=" * 60)
    
    print("""
DEPLOYING PARAMETER-EFFICIENT MODELS:

OPTION 1: Load Base + LoRA (Memory-Conscious)

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "lora_adapter_name",
    device_map="auto"
)

# Use model
output = model.generate(input_ids)
```

Pros:
- Flexibility (can swap LoRA)
- Memory efficient

Cons:
- Requires base model
- Merging/loading overhead


OPTION 2: Merge & Deploy (Speed-Optimized)

```python
# After training, merge
merged_model = peft_model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("merged_model")

# Load just merged model (no peft needed)
from transformers import AutoModel
model = AutoModel.from_pretrained("merged_model")
```

Pros:
- Single model file
- Fastest inference
- No peft dependency

Cons:
- Can't swap LoRA easily
- More storage per task


OPTION 3: Multi-Task Deployment

```python
base_model = load_base_model("gpt2")

tasks = {
    "sentiment": load_lora("sentiment_lora"),
    "translation": load_lora("translation_lora"),
    "summarization": load_lora("summarization_lora"),
}

def predict(text, task_name):
    lora = tasks[task_name]
    model.load_adapter(lora)  # Switch task
    return model.generate(text)
```


DEPLOYMENT OPTIONS:

1. Cloud (AWS, GCP, Azure)
   - AWS SageMaker
   - Use merged model
   - Auto-scaling

2. On-Premise
   - Deploy merged model
   - Standard model serving (TensorFlow Serving, etc)

3. Edge Device
   - Quantize merged model
   - LoRA very efficient for edge

4. API Service
   - FastAPI + Uvicorn
   - Load base + LoRA in memory
   - Serve via REST/gRPC


MONITORING:

Track:
- Inference latency
- Model drift (retrain if data changes)
- Task-specific metrics
- Resource usage


UPDATING MODELS:

For new data:
1. Retrain LoRA with new data
2. Save new LoRA weights
3. A/B test old vs new
4. Deploy new version
5. Keep rollback capability

Multiple LoRA versions easy:
- Model v1.0 (original)
- Model v1.1 (tweaked)
- Model v1.2 (retrained)

Different from full fine-tuning:
- Would need 340MB × 3 = 1GB storage
- With LoRA: 15MB × 3 = 45MB storage
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PARAMETER-EFFICIENT FINE-TUNING EXAMPLES")
    print("=" * 60)
    
    # Example 1: Challenges
    example_1_finetuning_challenges()
    
    # Example 2: LoRA
    example_2_lora_explained()
    
    # Example 3: Prefix Tuning
    example_3_prefix_tuning()
    
    # Example 4: Adapters
    example_4_adapters()
    
    # Example 5: Comparison
    example_5_comparative_analysis()
    
    # Example 6: LoRA Implementation
    example_6_lora_implementation()
    
    # Example 7: Training Tips
    example_7_training_considerations()
    
    # Example 8: Deployment
    example_8_production_deployment()
    
    print("\n" + "=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print("""
KEY TAKEAWAYS:

1. Traditional fine-tuning: Expensive, slow, impractical for large models
2. LoRA: Best balance of quality, speed, and efficiency (~0.5-5% parameters)
3. Adapters: Best modularity, separate adapters per task
4. Prefix Tuning: Minimal parameters, good for task adaptation
5. Pick based on constraints: Quality vs efficiency vs speed
6. LoRA recommended for most use cases
7. Easy deployment: Merge weights for inference

QUICK START:

1. Install peft: pip install peft
2. Wrap model with LoRA config
3. Train normally
4. Save model (15MB instead of 340MB)
5. Deploy merged model or load base+LoRA

NEXT STEPS:
- Try LoRA on your favorite model
- Experiment with rank sizes
- Compare quality vs training time
- Deploy in production
- Monitor performance
    """)


if __name__ == "__main__":
    main()
