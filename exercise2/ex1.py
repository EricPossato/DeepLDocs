import numpy as np
import matplotlib.pyplot as plt

def generate_and_plot():
    # Parameters for Class 0
    mean_class0 = [1.5, 1.5]
    cov_class0 = [[0.5, 0], [0, 0.5]]

    # Parameters for Class 1
    mean_class1 = [5, 5]
    cov_class1 = [[0.5, 0], [0, 0.5]]

    # Fix random seed for reproducibility
    np.random.seed(42)

    # Generate 1000 samples per class
    class0_samples = np.random.multivariate_normal(mean_class0, cov_class0, 1000)
    class1_samples = np.random.multivariate_normal(mean_class1, cov_class1, 1000)

    # Plot the two classes
    plt.figure(figsize=(8, 6))
    plt.scatter(class0_samples[:, 0], class0_samples[:, 1], 
                c='blue', alpha=0.5, label="Class 0")
    plt.scatter(class1_samples[:, 0], class1_samples[:, 1], 
                c='red', alpha=0.5, label="Class 1")

    plt.title("2D Data Points from Two Classes (Multivariate Normal)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True)
    #plt.show()
    # Save the image as a png file
    plt.savefig("exercise2_ex1_plot.png")
    # Return values for reuse
    return class0_samples, class1_samples

# Use it like this:
class0, class1 = generate_and_plot()
print("Class 0 shape:", class0)
# # Example: Combine and create labels
# X = np.vstack((class0, class1))
# y = np.hstack((-1 * np.ones(len(class0)), np.ones(len(class1))))

# print("X shape:", X.shape)
# print("y shape:", y.shape)
