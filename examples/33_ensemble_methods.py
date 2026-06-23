#!/usr/bin/env python3
"""
Example 33: Ensemble Methods

Learn how to combine multiple models for better performance.
Concepts: Model averaging, voting, stacking, boosting, uncertainty quantification
"""

import sys
from pathlib import Path


def example_1_ensemble_concepts():
    """Explain why ensembles work."""
    print("=" * 60)
    print("EXAMPLE 1: Why Ensembles Work?")
    print("=" * 60)
    
    print("""
ENSEMBLE PRINCIPLE: "Wisdom of Crowds"

Single Expert:
- Can make mistakes
- Limited perspective
- Single decision

Multiple Experts:
- Different perspectives
- Errors cancel out (if independent)
- Better collective decision


VOTING EXAMPLE:

Task: Predict if movie is good or bad

Single Model (BERT):
  Prediction: Good (70% confidence)
  
Ensemble of 5 models:
  BERT:        Good (70%)
  RoBERTa:     Good (75%)
  DistilBERT:  Bad  (55%)
  GPT-2:       Good (68%)
  T5:          Good (72%)
  
  Ensemble: 4/5 vote "Good" → High confidence!


MATHEMATICAL INSIGHT:

Error function: L = Σ (prediction_i - true_value)²

If errors are independent:
- Individual model error: e
- Ensemble error: e / √n

With 4 independent models:
- Individual: error = 4%
- Ensemble: error = 4% / 2 = 2%

Ensemble ERROR = Individual Error / √(number of models)


CONDITIONS FOR GOOD ENSEMBLES:

1. DIVERSITY
   ✓ Different model types (BERT, RoBERTa, GPT)
   ✓ Different training data
   ✓ Different hyperparameters
   
   ✗ Same models → No benefit
   ✗ Perfect correlation → No diversity

2. ACCURACY
   ✓ Each model better than random
   ✓ Models should be reasonably good
   
   ✗ Bad models together = worse

3. INDEPENDENCE
   ✓ Errors uncorrelated
   ✓ Different architectures help
   ✓ Different datasets help
   
   ✗ Identical training → Correlated errors


ACCURACY vs DIVERSITY TRADE-OFF:

High Accuracy, Low Diversity:
- All models very good (95% each)
- Make same correct predictions
- No error cancellation
- Ensemble improves little

Low Accuracy, High Diversity:
- Models different (70-80% each)
- Very different predictions
- High error cancellation
- Ensemble improves a lot

OPTIMAL: Accurate + Diverse
- Good models (90%+ each)
- Different architectures
- Trained differently
- Maximum improvement
    """)


def example_2_averaging_ensemble():
    """Implement simple averaging ensemble."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Simple Averaging Ensemble")
    print("=" * 60)
    
    print("""
AVERAGING ENSEMBLE: Average predictions from all models

PROCESS:

Step 1: Get predictions from all models
Model 1: [0.7, 0.2, 0.1]
Model 2: [0.6, 0.3, 0.1]
Model 3: [0.8, 0.1, 0.1]

Step 2: Average
Average: [(0.7+0.6+0.8)/3, (0.2+0.3+0.1)/3, (0.1+0.1+0.1)/3]
       = [0.70, 0.20, 0.10]

Step 3: Predict
argmax([0.70, 0.20, 0.10]) = Class 0


EXAMPLE CODE:

def averaging_ensemble(predictions):
    '''
    predictions: List of model outputs
                 Each output shape: (batch_size, num_classes)
    '''
    import numpy as np
    
    # Stack predictions
    stacked = np.array(predictions)  # (num_models, batch_size, num_classes)
    
    # Average across models (axis 0)
    ensemble_pred = np.mean(stacked, axis=0)
    
    return ensemble_pred


VARIANTS:

1. Simple Averaging
   Average prediction scores equally

2. Weighted Averaging
   Average = (w1×pred1 + w2×pred2 + w3×pred3) / (w1+w2+w3)
   
   Weights based on:
   - Model accuracy (better models get higher weight)
   - Model confidence
   - Model type
   
   Example:
   Weights = [0.5, 0.3, 0.2]  (Model 1 best)
   Average = (0.5×[0.7,0.2,0.1] + 0.3×[0.6,0.3,0.1] + 0.2×[0.8,0.1,0.1])
           = [0.70, 0.22, 0.08]

3. Log Averaging (for probabilities)
   - Average log-probabilities
   - Works better for extreme values

4. Geometric Averaging
   - Geometric mean instead of arithmetic
   - Less sensitive to outliers


PROPERTIES:

Pros:
✓ Simple to implement
✓ Works with any models
✓ No training required
✓ Fast
✓ Effective

Cons:
✗ All models contribute equally
✗ Bad models hurt ensemble
✗ Doesn't use information about model quality
    """)


def example_3_voting_ensemble():
    """Implement voting-based ensemble."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Voting Ensemble")
    print("=" * 60)
    
    print("""
VOTING ENSEMBLE: Models "vote" on prediction

PROCESS:

Task: Binary classification (0 or 1)

Model 1: Predicts 1
Model 2: Predicts 1
Model 3: Predicts 0
Model 4: Predicts 1
Model 5: Predicts 0

Voting: 3 votes for class 1, 2 votes for class 0
Ensemble Prediction: Class 1


EXAMPLE CODE:

def voting_ensemble(predictions):
    '''
    predictions: List of hard predictions
                 Each prediction shape: (batch_size,)
    '''
    import numpy as np
    
    stacked = np.array(predictions)  # (num_models, batch_size)
    
    # Mode along models (axis 0)
    from scipy.stats import mode
    ensemble_pred, _ = mode(stacked, axis=0)
    
    return ensemble_pred.flatten()


VOTING VARIANTS:

1. Majority Voting
   - Most common vote wins
   - Ties possible (even number of models)
   
   Example: [1, 1, 0, 1, 0] → Majority is 1

2. Weighted Voting
   - Better models get more votes
   
   Model 1 (90% acc): 2 votes for 1
   Model 2 (80% acc): 1 vote for 1
   Model 3 (75% acc): 1 vote for 0
   
   Total: 3 votes for 1, 1 vote for 0 → Predict 1

3. Unanimous Voting
   - All models must agree
   - Used when very high confidence needed
   - Many abstentions (send to human review)

4. Plurality Voting
   - Whatever gets most votes wins
   - Works even with multiple classes


PROPERTIES:

Pros:
✓ Simple
✓ Robust to bad predictions
✓ Works for classification
✓ Interpretable

Cons:
✗ Loses confidence information
✗ Not for regression
✗ Ties possible
✗ Discrete output (can't get probability)
    """)


def example_4_stacking_ensemble():
    """Explain stacking ensemble approach."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Stacking (Meta-Learner) Ensemble")
    print("=" * 60)
    
    print("""
STACKING: Use another model to combine predictions

INTUITION:
"Instead of averaging, learn how to best combine predictions"

TWO-LEVEL APPROACH:

Level 0 (Base Learners):
Model 1: BERT
Model 2: RoBERTa
Model 3: DistilBERT
(Train on training data)

Level 1 (Meta-Learner):
Classifier (often simple: Logistic Regression)
(Train on predictions of base models)

PROCESS:

Step 1: Train base models on training data
BERT trained on 80% of data
RoBERTa trained on 80% of data
DistilBERT trained on 80% of data

Step 2: Generate meta-features (on validation split)
Pass 20% through each base model
Get predictions as new features

For example:
Original: "This movie is great"
BERT output:     [0.7, 0.3]  (prob of positive, negative)
RoBERTa output:  [0.75, 0.25]
DistilBERT output: [0.65, 0.35]

Meta-features: [0.7, 0.3, 0.75, 0.25, 0.65, 0.35]

Step 3: Train meta-learner on meta-features
Logistic Regression learns:
"When BERT says 0.7 and RoBERTa says 0.75... trust them"
"When models disagree... weight by confidence"

Step 4: Predict with stacking
Test example:
BERT: [0.6, 0.4]
RoBERTa: [0.7, 0.3]
DistilBERT: [0.5, 0.5]

Meta-features: [0.6, 0.4, 0.7, 0.3, 0.5, 0.5]
Meta-learner: Predicts sentiment based on pattern


EXAMPLE CODE:

def stacking_ensemble(base_models, meta_model, X_train, X_val, y_train, y_val):
    '''
    Stacking 2-level ensemble
    '''
    # Step 1: Train base models
    for model in base_models:
        model.fit(X_train, y_train)
    
    # Step 2: Generate meta-features from validation
    meta_features = []
    for model in base_models:
        predictions = model.predict_proba(X_val)  # Get probabilities
        meta_features.append(predictions)
    
    # Stack: (n_samples, n_models × n_classes)
    X_meta = np.hstack(meta_features)
    
    # Step 3: Train meta-learner
    meta_model.fit(X_meta, y_val)
    
    return base_models, meta_model


ADVANTAGES:

✓ Learns optimal combination
✓ Can be more sophisticated
✓ Flexible (any meta-learner)
✓ Often better than averaging
✓ Data-driven approach

DISADVANTAGES:

✗ More complex
✗ Requires more computation
✗ Prone to overfitting (needs careful validation)
✗ Harder to debug
✗ Need held-out validation set


META-LEARNER CHOICES:

1. Logistic Regression (classification)
   - Simple, fast
   - Linear combination of base models
   
2. Neural Network
   - More complex
   - Learn non-linear combinations
   - Risk of overfitting
   
3. Gradient Boosting
   - Combines base predictions sequentially
   - learns relative importance

4. Another transformer
   - For NLP tasks
   - Most flexible but expensive
    """)


def example_5_boosting_ensemble():
    """Explain boosting approach."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Boosting Ensemble")
    print("=" * 60)
    
    print("""
BOOSTING: Sequentially build models, focusing on errors

CORE IDEA:
Train subsequent models to fix mistakes of previous models

ADABOOST PROCESS:

Initial: All samples have equal weight = 1/N

Iteration 1:
├─ Train Model 1 on all samples
├─ Model 1 makes mistakes on some samples
└─ Increase weight of misclassified samples

Iteration 2:
├─ Train Model 2 on weighted samples
├─ Harder: focuses on previously wrong samples
├─ Model 2 makes new mistakes
└─ Increase weight of newly misclassified samples

Iteration 3:
├─ Train Model 3 focusing even more
└─ ...continue...

For each sample:
weight₁ = 1/N (initial)
weight₂ = weight₁ × α¹ (increase if Model 1 wrong)
weight₃ = weight₂ × α² (increase if Model 2 wrong)


EXAMPLE:

Dataset: 10 samples
Initial weights: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

Model 1 training:
- 8 samples correct
- 2 samples wrong
- Weighted error: 0.2
- Update weights for wrong samples

Updated weights: [1, 1, 2, 2, 1, 1, 1, 1, 1, 1]
(samples 3 and 4 got higher weight)

Model 2 training:
- Focuses more on samples 3, 4
- Maybe gets them right but wrong on others
- Say samples 1, 5 now wrong
- Update weights: [2, 1, 1, 1, 2, 1, 1, 1, 1, 1]

Model 3 training:
- Focuses on samples 1, 5
- Gradually reduces overall error


WEIGHTED VOTING:

Final prediction = Σ (accuracy_i × model_i_prediction)

Model 1 accuracy: 80% → weight = log(1/0.2) = 1.61
Model 2 accuracy: 90% → weight = log(1/0.1) = 2.30
Model 3 accuracy: 85% → weight = log(1/0.15) = 1.90

Better models get higher weight in final vote


ADVANTAGES:

✓ Focuses on hard examples
✓ Often better than voting
✓ Reduced bias and variance
✓ Principled approach
✓ Sequential improvement

DISADVANTAGES:

✗ Requires correct labels for all iterations
✗ Sequential (slow to train)
✗ Sensitive to outliers
✗ Can overfit if too many iterations


VARIANTS:

1. AdaBoost
   - Adaptive boosting
   - Classical approach
   - Works well for classification

2. Gradient Boosting
   - Boost using gradients
   - More flexible
   - Works for regression too

3. XGBoost
   - Extreme Gradient Boosting
   - Optimized implementation
   - Popular in competitions

4. LightGBM
   - Fast gradient boosting
   - Efficient memory usage
    """)


def example_6_diversity_techniques():
    """Techniques to improve ensemble diversity."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Improving Ensemble Diversity")
    print("=" * 60)
    
    print("""
TO IMPROVE ENSEMBLE, IMPROVE DIVERSITY:

1. DIFFERENT MODELS

Use architectures:
✓ BERT (12 layers, 110M params)
✓ RoBERTa (24 layers, 355M params)
✓ ALBERT (smaller, lighter)
✓ DistilBERT (distilled version)
✓ GPT-2 (generative)
✓ T5 (encoder-decoder)

Why: Different architectures learn different patterns


2. DIFFERENT TRAINING DATA

Bagging approach:
- Train BERT on 80% of data (randomly sampled subset A)
- Train BERT on different 80% (randomly sampled subset B)
- Train BERT on different 80% (randomly sampled subset C)

Why: Sampling variance creates diversity
Result: Same architecture, different training = diversity


3. DIFFERENT HYPERPARAMETERS

Same architecture, different config:
- Model 1: learning_rate=5e-5, batch_size=32
- Model 2: learning_rate=1e-4, batch_size=16
- Model 3: learning_rate=2e-5, batch_size=64

Why: Different optimization paths → different models
Effect: Modest diversity


4. DIFFERENT INITIALIZATION

Same config, different random seeds:
- Model 1: seed=42
- Model 2: seed=43
- Model 3: seed=44

Why: Neural networks sensitive to initialization
Effect: Creates variation in learned weights


5. TASK-SPECIFIC DATA

For same task, use different perspectives:
- Sentiment: Use different review sources
- Translation: Train on different domains
- QA: Use different annotation teams


DIVERSITY MEASUREMENT:

1. Prediction Disagreement
   - For same sample, how often do they disagree?
   - Higher = more diversity
   
   Formula: disagreement = (# samples models disagree) / (# total samples)

2. Ensemble Margin
   - Difference between top 2 predictions
   - Larger margin = more confident
   
3. KL Divergence
   - How different are probability distributions?
   - Higher = more different

4. Correlation Analysis
   - Predictions of model pairs
   - Lower correlation = higher diversity


PRACTICAL CHECKLIST:

For good ensemble:
- [ ] Use 3-5 different base models
- [ ] Different architectures (BERT, RoBERTa, DistilBERT)
- [ ] Different training data or hyperparameters
- [ ] Measure disagreement (target: 20-30% for classification)
- [ ] Test ensemble on validation set
- [ ] Verify improvement over individual models
- [ ] Consider computational cost
- [ ] Monitor diversity during inference
    """)


def example_7_uncertainty_quantification():
    """Estimate confidence and uncertainty."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Uncertainty Quantification via Ensembles")
    print("=" * 60)
    
    print("""
ENSEMBLES GIVE UNCERTAINTY ESTIMATES:

SINGLE MODEL:
"Sentiment: Positive (82% confidence)"

ENSEMBLE:
"Sentiment: Positive"
Confidence sources:
- 5 models agree strongly: 4/5 say positive, 1/5 says negative
- Ensemble confidence: 80% (given disagreement)
- Expert agreement: 80%

UNCERTAINTY SOURCES:

1. ALEATORIC UNCERTAINTY (Data Uncertainty)
   - Noise in the data itself
   - Can't reduce by collecting more models

2. EPISTEMIC UNCERTAINTY (Model Uncertainty)
   - Model doesn't know
   - Can reduce by ensemble
   - Reduced by different models

Ensemble estimates EPISTEMIC uncertainty


CONFIDENCE MEASURES:

1. AGREEMENT RATIO
   - What fraction of models agree?
   
   Example: 4/5 models vote positive
   Agreement: 80%
   Interpretation: High confidence

2. ENTROPY
   - Spread of predictions
   - High entropy = high uncertainty
   
   Votes: [4, 1] for classes [pos, neg]
   P(pos) = 0.8, P(neg) = 0.2
   Entropy = -0.8×log(0.8) - 0.2×log(0.2) = 0.72
   
   Low entropy = certain
   High entropy = uncertain

3. VARIANCE
   - Model disagreement magnitude
   
   Predictions: [0.8, 0.7, 0.85, 0.75, 0.9]
   Mean: 0.8
   Variance: 0.0050 (very small = confident)
   
   Predictions: [0.5, 0.8, 0.3, 0.9, 0.2]
   Mean: 0.56
   Variance: 0.1164 (larger = uncertain)

4. PREDICTIVE INTERVAL
   - Range of predictions
   
   Min prediction: 0.2
   Max prediction: 0.9
   Interval: [0.2, 0.9] (wide = uncertain)


EXAMPLE:

Sample 1: "I love this!"
- Model 1: positive (0.95)
- Model 2: positive (0.93)
- Model 3: positive (0.91)
- Model 4: positive (0.94)
- Model 5: positive (0.92)

Ensemble: Positive (0.93 ± 0.01)
Confidence: Very high ✓
Action: Use prediction confidently

Sample 2: "The movie was okay."
- Model 1: positive (0.55)
- Model 2: negative (0.48)
- Model 3: positive (0.51)
- Model 4: neutral (0.50)  # Hypothetical
- Model 5: negative (0.49)

Ensemble: Uncertain (0.50 ± 0.03)
Confidence: Low ✗
Action: Ask for human review / get more context


USE CASES:

1. Reject Option
   - High uncertainty → don't predict
   - Ask for human decision

2. Active Learning
   - Uncertain samples → ask human to label
   - Label most informative samples first

3. Anomaly Detection
   - Out-of-distribution → high uncertainty
   - Use uncertainty to detect anomalies

4. Mining Hard Examples
   - Find samples ensemble struggles with
   - Collect more data like them

5. Cost-Sensitive Decisions
   - High-cost decisions → only if high confidence
   - Low-cost decisions → can use uncertain predictions
    """)


def example_8_practical_considerations():
    """Practical considerations for ensembles."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Practical Considerations")
    print("=" * 60)
    
    print("""
COMPUTATIONAL COST:

Ensemble of 5 BERT models:
- Single model: 4 hours training, 100ms inference
- Ensemble training: 20 hours (5× more)
- Ensemble inference: 500ms (5× slower)

Trade-offs:
Speed:        Single > Ensemble
Accuracy:     Ensemble >> Single
Cost:         Ensemble > Single


TRAINING TIME ANALYSIS:

Models to train:
- 1 model: T time
- 5 models: 5T time (can parallelize to reduce)
- Use different seeds or data subsets
- Can train on different GPUs simultaneously

Cost estimate:
Single GPU, 5 models: 5×training_time
5 GPUs, 5 models: 1×training_time (parallel)


INFERENCE TIME ANALYSIS:

Single model: 100ms
5 models: 500ms sequential
Averaging: 500ms + 10ms = 510ms
Bottleneck: Model inference

Optimization:
- Use quantized models
- Use distillation (smaller models)
- Use batching
- Use GPU optimization


TRADE-OFFS TO CONSIDER:

When to use ensemble:
✓ Accuracy is critical
✓ Budget available
✓ Can parallelize training
✓ Inference time not critical

When NOT to use:
✗ Speed critical (latency requirements)
✗ Minimal budget
✗ Accuracy good enough with single model
✗ Can't store multiple models


COST-BENEFIT ANALYSIS:

Example: Sentiment classifier for reviews

Single BERT:
- Accuracy: 92%
- Cost: $10
- Speed: 100ms
- False positives: 800/10000

5-model ensemble:
- Accuracy: 96%
- Cost: $50
- Speed: 500ms
- False positives: 400/10000

If each false positive costs $100:
- Single: 800 × $100 = $80,000
- Ensemble: 400 × $100 = $40,000
- Savings: $40,000
- Net benefit: $40,000 - $40 = $39,960 ✓

Ensemble is worth it!


DEPLOYMENT STRATEGIES:

1. All Models Online
   - All models always running
   - Best accuracy
   - Highest cost

2. Single Model Primary + Ensemble Fallback
   - Fast model normally
   - Ensemble only for uncertain cases
   - Balanced speed/accuracy

3. Offline Ensemble
   - Train ensemble offline
   - Merge to single model via distillation
   - Fast inference, good accuracy


RECOMMENDATION:

For most projects:
- Start with single good model
- If accuracy insufficient: try ensemble
- Use 3-5 diverse models
- Avoid >10 models (diminishing returns)
- Consider distillation to compress
- Monitor inference cost

For high-stakes (medical, legal):
- Use ensemble + uncertainty estimates
- Reject low-confidence predictions
- Use human-in-the-loop review
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ENSEMBLE METHODS EXAMPLES")
    print("=" * 60)
    
    # Example 1: Concepts
    example_1_ensemble_concepts()
    
    # Example 2: Averaging
    example_2_averaging_ensemble()
    
    # Example 3: Voting
    example_3_voting_ensemble()
    
    # Example 4: Stacking
    example_4_stacking_ensemble()
    
    # Example 5: Boosting
    example_5_boosting_ensemble()
    
    # Example 6: Diversity
    example_6_diversity_techniques()
    
    # Example 7: Uncertainty
    example_7_uncertainty_quantification()
    
    # Example 8: Practical
    example_8_practical_considerations()
    
    print("\n" + "=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print("""
KEY TAKEAWAYS:

1. Ensembles combine models for better accuracy
2. Key: Diversity + Accuracy + Independence
3. Averaging: Simple, effective, no training
4. Voting: Good for classification
5. Stacking: Learns optimal combination
6. Boosting: Sequentially improves
7. Uncertainty: Estimate confidence via disagreement
8. Cost: Trade-off speed vs accuracy

ENSEMBLE STRATEGIES:

Quick Win (Simple Averaging):
- Train 3-5 models independently
- Average their outputs
- +1-2% accuracy, minimal cost

Optimal (Balanced):
- Diverse models (BERT, RoBERTa, DistilBERT)
- Weighted averaging or stacking
- +2-4% accuracy, moderate cost

Production (High Stakes):
- Ensemble + uncertainty quantification
- Reject option for uncertain cases
- Human review for edge cases
- +3-5% accuracy, higher cost

NEXT STEPS:
- Start with simple averaging on existing models
- Measure improvement
- Measure inference time cost
- Decide if worth deploying
- Consider distillation if needed
- Monitor real-world performance
    """)


if __name__ == "__main__":
    main()
