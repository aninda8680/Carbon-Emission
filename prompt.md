Create a 15-slide academic presentation for a college audience. The style should be professional, confident, and technical yet accessible. The primary focus must be our main achievement: a massive 5.4x training speedup with absolutely zero accuracy loss using LightGBM and Feature Selection.

Please follow this exact outline and use the provided text for each slide:

### Slide 1: Title
Carbon-Efficient Machine Learning for Sustainable Model Training
Mini Project Presentation

Presented By:
* Purabi Malakar (UG/02/BTCSE/2023/067)
* Aninda Debta (UG/02/BTCSE/2023/103)

Department of Computer Science & Engineering
Adamas University, Kolkata
Mentor : Dr. Debdutta Pal

### Slide 2: Introduction
* Artificial Intelligence and Machine Learning are widely used in modern applications and are transforming modern industries.
* Training these machine learning models requires high computational power.
* Increased computation leads to high energy consumption and carbon emissions.
* Traditional ML focuses mainly on accuracy, often ignoring sustainability.
* Our Focus: This project focuses on developing a carbon-efficient machine learning framework that reduces environmental impact while maintaining high model performance.

### Slide 3: Problem Definition
* Traditional machine learning models consume significant computational resources.
* Higher energy consumption increases environmental impact and carbon emissions.
* Many traditional models perform redundant computations.
* The lack of sustainability-focused metrics in machine learning evaluation creates a blind spot.
* The Need: Efficient and environmentally responsible AI systems that reduce environmental impact without affecting accuracy.

### Slide 4: Objectives
* Develop a carbon-efficient ML framework.
* Reduce training time and computational cost.
* Maintain high predictive accuracy.
* Implement optimization techniques for sustainable AI.
* Compare baseline and optimized models.
* Introduce the Carbon Efficiency Gain Index (CEGI).

### Slide 5: Dataset Details
Dataset Name: Child Health West Bengal Dataset
* Dataset Information:
  * Total Samples: 1000
  * Features: 7
  * Target Variable: Multiclass Classification (Normal, Moderate, Severe)
* Preprocessing Steps:
  * Missing Value Handling
  * Label Encoding
  * Stratified Train-Test Split (80:20)
  * Float64 to Float32 Memory Optimization

### Slide 6: System Description
Workflow:
Dataset Collection
↓
Data Preprocessing
↓
Memory Optimization (Float32 Conversion)
↓
Baseline Model Training (RandomForest)
↓
Optimized Model Training (LightGBM + Feature Selection / PCA)
↓
CO₂ Emission Tracking (CodeCarbon)
↓
Performance Comparison
↓
Result Visualization

### Slide 7: Models Used
* Baseline Model: RandomForest Classifier
  * Used as a benchmark model.
  * High accuracy, but higher computational cost.
* Optimized Models:
  * LightGBM + Feature Selection: Faster training, lower computation cost, reduced feature complexity.
  * PCA + LightGBM: Balanced performance, reduced dimensions.
  * Minimal Model: Maximum efficiency, suitable for edge devices.

### Slide 8: The 8 Optimization Techniques
To achieve carbon efficiency, we implemented 8 specific strategies:
1. Memory Optimization: Float64 to Float32 conversion (50% memory reduction).
2. Feature Selection: Removed redundant features using Mutual Information.
3. LightGBM Algorithm: Switched to a highly efficient tree-based algorithm.
4. Early Stopping: Prevented wasted computation during training.
5. PCA Reduction: Compressed feature dimensions.
6. Hyperparameter Tuning: Energy-aware parameter selection.
7. Batch Optimization: Vectorized operations for speed.
8. Model Simplification: Progressive reduction of complexity.

### Slide 9: Novel Innovation: The CEGI Metric
* Problem: Standard metrics only measure accuracy or speed, not environmental impact.
* Solution: We created the Carbon Efficiency Gain Index (CEGI).
* Formula: CEGI = Accuracy / CO₂ Emissions
* Significance:
  * Allows direct comparison of sustainability vs. performance.
  * Higher CEGI means the model achieves high accuracy while polluting less.
  * Our LightGBM + FS model achieved a massive CEGI score of 119,004.

### Slide 10: Output
* Baseline Model (RandomForest)
  * Accuracy: 100%
  * Training Time: 0.27 sec
  * CO₂ Emission: 0.000008 kg
* Optimized Model (LightGBM + Feature Selection)
  * Accuracy: 100%
  * Training Time: 0.05 sec
  * Result: 5.4× faster performance with 0% accuracy loss
* Other Models
  * Balanced Model Accuracy: 98.5%
  * Minimal Model Accuracy: 98.5% (~13.5× faster execution)

### Slide 11: Visualizing Efficiency: Time & Emissions
(Please instruct me to insert Graph Image Here: `emissions_time_comparison.png`)
* Key Takeaways:
  * The graph demonstrates the stark contrast in training time between the Baseline and Optimized models.
  * Training time dropped dramatically from 0.27s to 0.05s.
  * This proves that intelligent algorithm selection (LightGBM) significantly reduces processing duration, leading directly to lower power consumption.

### Slide 12: Visualizing Performance: CEGI Score
(Please instruct me to insert Graph Image Here: `cegi_comparison.png`)
* Key Takeaways:
  * Visual proof of our optimization success.
  * The LightGBM + Feature Selection model towers over the baseline in terms of pure efficiency.
  * Demonstrates that our optimized model is the most environmentally friendly choice without sacrificing reliability.

### Slide 13: Three-Tier Optimization Approach
We developed three practical deployment tiers based on our findings:
1. Tier 1: High Accuracy (Production-Ready)
  * LightGBM + Feature Selection (100% Accuracy, 5.4x Faster)
2. Tier 2: Balanced (General Use)
  * PCA + LightGBM (98.5% Accuracy, 4.5x Faster)
3. Tier 3: Maximum Speed (Prototyping/Edge Devices)
  * Minimal Complexity (98.5% Accuracy, 13.5x Faster)

### Slide 14: Analysis and Conclusion
Analysis
* Optimization reduced training time significantly (5.4x speedup).
* Feature selection reduced computational complexity by 14%.
* Carbon efficiency improved substantially as shown by our CEGI metric.
* High model accuracy (100%) was perfectly maintained.

Conclusion
* Sustainable machine learning is highly practical.
* Green AI effectively reduces environmental impact.
* Optimization improves efficiency without sacrificing performance.

### Slide 15: References
* Strubell et al. (2019) – Energy and Policy Considerations for Deep Learning in NLP
* Patterson et al. (2021) – Carbon Emissions and Large Neural Network Training
* Schwartz et al. (2020) – Green AI
* Ke et al. (2017) – LightGBM
* CodeCarbon Documentation
* Scikit-learn Documentation


