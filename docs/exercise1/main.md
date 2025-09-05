
!!! success inline end "Deadline and Submission"

    :date: 05.sep (friday)
    
    :clock1: Commits until 23:59

    :material-account: Individual

    :simple-github: Submission the GitHub Pages' Link (yes, **only** the link for pages) via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.


**Activity: Data Preparation and Analysis for Neural Networks**

This activity is designed to test your skills in generating synthetic datasets, handling real-world data challenges, and preparing data to be fed into **neural networks**.

***

## Exercise 1

### **Exploring Class Separability in 2D**

Understanding how data is distributed is the first step before designing a network architecture. In this exercise, you will generate and visualize a two-dimensional dataset to explore how data distribution affects the complexity of the decision boundaries a neural network would need to learn.

### **Instructions**

1.  **Generate the Data:** Create a synthetic dataset with a total of 400 samples, divided equally among 4 classes (100 samples each). Use a Gaussian distribution to generate the points for each class based on the following parameters:
    * **Class 0:** Mean = $[2, 3]$, Standard Deviation = $[0.8, 2.5]$
    * **Class 1:** Mean = $[5, 6]$, Standard Deviation = $[1.2, 1.9]$
    * **Class 2:** Mean = $[8, 1]$, Standard Deviation = $[0.9, 0.9]$
    * **Class 3:** Mean = $[15, 4]$, Standard Deviation = $[0.5, 2.0]$
1.  **Plot the Data:** Create a 2D scatter plot showing all the data points. Use a different color for each class to make them distinguishable.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Number of samples per class
n_samples = 100  

# Class parameters: (mean, std)
class_params = {
    0: {"mean": [2, 3], "std": [0.8, 2.5]},
    1: {"mean": [5, 6], "std": [1.2, 1.9]},
    2: {"mean": [8, 1], "std": [0.9, 0.9]},
    3: {"mean": [15, 4], "std": [0.5, 2.0]}
}

# Store all samples
data = []
labels = []

for label, params in class_params.items():
    mean = params["mean"]
    std = params["std"]
    
    # Generate Gaussian distributed samples
    x = np.random.normal(mean[0], std[0], n_samples)
    y = np.random.normal(mean[1], std[1], n_samples)
    
    # Stack into dataset
    samples = np.column_stack((x, y))
    data.append(samples)
    labels.extend([label] * n_samples)

# Combine into full dataset
data = np.vstack(data)
labels = np.array(labels)

# Put into a DataFrame
df = pd.DataFrame(data, columns=["x1", "x2"])
df["class"] = labels

# Visualization
plt.figure(figsize=(8,6))
for label in class_params.keys():
    subset = df[df["class"] == label]
    plt.scatter(subset["x1"], subset["x2"], label=f"Class {label}", alpha=0.6)

plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Synthetic Gaussian Dataset (4 classes)")
# Save image
plt.savefig("synthetic_gaussian_dataset.png")
```

    ![Synthetic Gaussian Dataset](synthetic_gaussian_dataset.png)

1.  **Analyze and Draw Boundaries:**
    1. Examine the scatter plot carefully. Describe the distribution and overlap of the four classes.
        - Class 0 has a spread across most of the vertical axis and and is between 0.0 and 4.0 horizontally. Slightly overlaps Class 1

        - Class 0 has a diagonal spread, between 2 and 10 vertically as well as horizontally. Slightly overlaps Class 0 in a few points and Class 2 in fewer points.

        - Class 2 is very concentrated and relatvely isolated from other classes.

        - Class 3 is completely isolated, spread vertically along and the furthest in the x axis.

    1. Based on your visual inspection, could a simple, linear boundary separate all classes?
        - A linear boundary is capable of mostly separating all classes (with some missclassified points), but because of the overlaps between class 0 and 1, a perfect and clean linear separation would fail.
    1. On your plot, sketch the decision boundaries that you think a trained neural network might learn to separate these classes.

        - By adding the following code to the earlier plot, we can approximate some separations between classes
        ```python
        # three hand-drawn separators 
        plt.plot([6.0, 0.6], [-2.0, 10.0], linestyle="--", linewidth=2, color="k", label="approx. boundary")  # between 0 & 1
        plt.plot([4.0, 12.0], [2.0, 5.0], linestyle="--", linewidth=2, color="k")                            # between 1 & 2
        plt.plot([12.0, 12.0], [-1, 9], linestyle="--", linewidth=2, color="k")                             # isolate class 3
        ```
        ![Synthetic Gaussian Dataset with Boundaries](synthetic_gaussian_dataset_with_boundaries.png)

***

## Exercise 2

### **Non-Linearity in Higher Dimensions**

Simple neural networks (like a Perceptron) can only learn linear boundaries. Deep networks excel when data is not linearly separable. This exercise challenges you to create and visualize such a dataset.

### **Instructions**

1.  **Generate the Data:** Create a dataset with 500 samples for Class A and 500 samples for Class B. Use a multivariate normal distribution with the following parameters:

    * **Class A:**

        Mean vector:

        $$\mu_A = [0, 0, 0, 0, 0]$$

        Covariance matrix:

        $$
        \Sigma_A = \begin{pmatrix}
        1.0 & 0.8 & 0.1 & 0.0 & 0.0 \\
        0.8 & 1.0 & 0.3 & 0.0 & 0.0 \\
        0.1 & 0.3 & 1.0 & 0.5 & 0.0 \\
        0.0 & 0.0 & 0.5 & 1.0 & 0.2 \\
        0.0 & 0.0 & 0.0 & 0.2 & 1.0
        \end{pmatrix}
        $$

    * **Class B:**

        Mean vector:
            
        $$\mu_B = [1.5, 1.5, 1.5, 1.5, 1.5]$$
        
        Covariance matrix:

        $$
        \Sigma_B = \begin{pmatrix}
        1.5 & -0.7 & 0.2 & 0.0 & 0.0 \\
        -0.7 & 1.5 & 0.4 & 0.0 & 0.0 \\
        0.2 & 0.4 & 1.5 & 0.6 & 0.0 \\
        0.0 & 0.0 & 0.6 & 1.5 & 0.3 \\
        0.0 & 0.0 & 0.0 & 0.3 & 1.5
        \end{pmatrix}
        $$

1.  **Visualize the Data:** Since you cannot directly plot a 5D graph, you must reduce its dimensionality.
    * Use a technique like **Principal Component Analysis (PCA)** to project the 5D data down to 2 dimensions.
    * Create a scatter plot of this 2D representation, coloring the points by their class (A or B).
1.  **Analyze the Plots:**
    1. Based on your 2D projection, describe the relationship between the two classes.
    1. Discuss the **linear separability** of the data. Explain why this type of data structure poses a challenge for simple linear models and would likely require a multi-layer neural network with non-linear activation functions to be classified accurately.

***

## Exercise 3

### **Preparing Real-World Data for a Neural Network**

This exercise uses a real dataset from Kaggle. Your task is to perform the necessary preprocessing to make it suitable for a neural network that uses the hyperbolic tangent (`tanh`) activation function in its hidden layers.

### **Instructions**

1.  **Get the Data:** Download the [**Spaceship Titanic**](https://www.kaggle.com/competitions/spaceship-titanic){:target="_blank"} dataset from Kaggle.
2.  **Describe the Data:**
    * Briefly describe the dataset's objective (i.e., what does the `Transported` column represent?).
    * List the features and identify which are **numerical** (e.g., `Age`, `RoomService`) and which are **categorical** (e.g., `HomePlanet`, `Destination`).
    * Investigate the dataset for **missing values**. Which columns have them, and how many?
3.  **Preprocess the Data:** Your goal is to clean and transform the data so it can be fed into a neural network. The `tanh` activation function produces outputs in the range `[-1, 1]`, so your input data should be scaled appropriately for stable training.
    * **Handle Missing Data:** Devise and implement a strategy to handle the missing values in all the affected columns. Justify your choices.
    * **Encode Categorical Features:** Convert categorical columns like `HomePlanet`, `CryoSleep`, and `Destination` into a numerical format. One-hot encoding is a good choice.
    * **Normalize/Standardize Numerical Features:** Scale the numerical columns (e.g., `Age`, `RoomService`, etc.). Since the `tanh` activation function is centered at zero and outputs values in `[-1, 1]`, **Standardization** (to mean 0, std 1) or **Normalization** to a `[-1, 1]` range are excellent choices. Implement one and explain why it is a good practice for training neural networks with this activation function.
4.  **Visualize the Results:**
    * Create histograms for one or two numerical features (like `FoodCourt` or `Age`) **before** and **after** scaling to show the effect of your transformation.

***

## **Evaluation Criteria**

The deliverable for this activity consists of a **report** that includes:

1. A brief description of your approach to each exercise.
1. The code used to generate the datasets, preprocess the data, and create the visualizations. With comments explaining each step.
1. The plots and visualizations requested in each exercise.
1. Your analysis and answers to the questions posed in each exercise.

**Important Notes:**

- The deliverable must be submitted in the format specified: **GitHub Pages**. **No other formats will be accepted.** - there exists a template for the course that you can use to create your GitHub Pages - [template](https://hsandmann.github.io/documentation.template/){target='_blank'};

- There is a **strict policy against plagiarism**. Any form of plagiarism will result in a zero grade for the activity and may lead to further disciplinary actions as per the university's academic integrity policies;

- **The deadline for each activity is not extended**, and it is expected that you complete them within the timeframe provided in the course schedule - **NO EXCEPTIONS** will be made for late submissions.

- **AI Collaboration is allowed**, but each student **MUST UNDERSTAND** and be able to explain all parts of the code and analysis submitted. Any use of AI tools must be properly cited in your report. ^^**ORAL EXAMS**^^ may require you to explain your work in detail.

- All deliverables for individual activities should be submitted through the course platform [insper.blackboard.com](http://insper.blackboard.com/){:target="_blank"}.

**Grade Criteria:**

**Exercise 1 (3 points):**

| Criteria | Description |
|:--------:|-------------|
| **1 pt** | Data is generated correctly and visualized in a clear scatter plot with proper labels and colors. |
| **2 pts** | The analysis of class separability is accurate, and the proposed decision boundaries are logical and well-explained in the context of what a network would learn. |

**Exercise 2 (3 points):**

| Criteria | Description |
|:--------:|-------------|
| **1 pt** | Data is generated correctly using the specified multivariate parameters. |
| **1 pt** | Dimensionality reduction is applied correctly, and the resulting 2D projection is clearly plotted. |
| **1 pt** | The analysis correctly identifies the non-linear relationship and explains why a neural network would be a suitable model. |

**Exercise 3 (4 points):**

| Criteria | Description |
|:--------:|-------------|
| **1 pt** | The data is correctly loaded, and its characteristics are accurately described. |
| **2 pts** | All preprocessing steps (handling missing data, encoding, and appropriate feature scaling for `tanh`) are implemented correctly and with clear justification for a neural network context. |
| **1 pt** | Visualizations effectively demonstrate the impact of the data preprocessing. |
