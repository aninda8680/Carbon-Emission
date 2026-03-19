# Step-by-Step Guide: Reducing Carbon Emissions in Machine Learning

This guide outlines exactly how we will complete Phase 1 and Phase 2 of our project. We will train two models on the `child_health_westbengal_1000.csv` dataset and measure the carbon footprint of each.

## Prerequisites
Before writing any code, open your terminal (command prompt) and install these required tools:
```bash
pip install pandas scikit-learn codecarbon lightgbm optuna
```

---

## Phase 1: Establish the Baseline (The "Before" Picture)
**Goal:** Train a standard, heavy model without any optimizations to establish our worst-case (baseline) carbon footprint.

### Step 1: Load and Prepare the Data
**Tool:** `pandas` and `scikit-learn`
*   **What it does:** Reads your `.csv` file and gets it ready for the computer to understand.
*   **Action:** 
    *   Load `child_health_westbengal_1000.csv`.
    *   Clean up any missing data.
    *   Convert text columns into numbers (machine learning models only understand math).
    *   Split the data into two parts: training data (to teach the model) and testing data (to check its work).

### Step 2: Set up the Carbon Tracker
**Tool:** `CodeCarbon`
*   **What it does:** Runs quietly in the background and measures how much electricity your computer is using while the code runs.
*   **Action:** Right before you start training the model, you tell `CodeCarbon` to start recording.

### Step 3: Train a Heavy Machine Learning Model
**Tool:** `scikit-learn` (Algorithm: `RandomForestClassifier` or `MLPClassifier`)
*   **What it does:** We intentionally choose an algorithm that does a lot of math and takes longer to run.
*   **Action:** We feed our prepared data into this heavy model so it can learn the patterns. We leave all settings on their standard defaults.

### Step 4: Stop Recording and Save Results
**Tool:** `CodeCarbon`
*   **What it does:** Halts the electricity measurement and saves a report.
*   **Action:** The moment the model finishes training, you tell the tracker to stop. CodeCarbon will automatically create an `emissions.csv` file showing you exactly how many kg of CO2 were produced and how long it took.

---



## Phase 2: Algorithmic Optimization (The "After" Picture)
**Goal:** Apply clever techniques to make the training process faster and more energy-efficient, proving that we can reduce emissions.

### Step 1: Make the Data "Lighter" (Downcasting)
**Tool:** `pandas`
*   **What it does:** Reduces the amount of computer memory the data takes up.
*   **Action:** We convert heavy 64-bit numbers into lighter 32-bit numbers. This means the computer processor has to do less work to read the data, saving electricity.

### Step 2: Choose a "Green" Algorithm
**Tool:** `LightGBM`
*   **What it does:** A specialized machine learning algorithm built specifically to be extremely fast and use very little memory.
*   **Action:** We replace the heavy "Random Forest" from Phase 1 with LightGBM.

### Step 3: Implement "Early Stopping"
**Tool:** Built into `LightGBM`
*   **What it does:** Stops the model from continuing to train if it isn't learning anything new.
*   **Action:** We tell LightGBM, "If you go through 10 rounds of training and your accuracy doesn't go up, just turn off immediately." This saves massive amounts of wasted energy.

### Step 4: Energy-Efficient Hyperparameter Tuning
**Tool:** `Optuna`
*   **What it does:** Finds the best settings for your model without brute-forcing it.
*   **Action:** Normally, finding the best settings requires training the model hundreds of times (which burns lots of carbon). Optuna uses smart math (probability) to guess the best settings, finding the perfect model in far fewer attempts.

### Step 5: Test and Measure the Optimized Process
**Tool:** `CodeCarbon`
*   **What it does:** Tracks the emissions again, just like in Phase 1.
*   **Action:** We start the tracker, run our new optimized Lightfoot model with Early Stopping, and stop the tracker. 

### Final Step: The Comparison
At the end, we open the `emissions.csv` file and compare Phase 1 and Phase 2. We will look at:
1.  **Emissions:** Did the kg of CO2 drop?
2.  **Time:** Did the training time decrease?
3.  **Accuracy:** Did the model still correctly predict the outcomes? (The goal is a huge drop in CO2 with almost no drop in accuracy).