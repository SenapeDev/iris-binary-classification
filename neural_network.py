import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Constants
FILE_NAME = "iris.data"
FEATURES = 2
EPOCHS = 100
ETA = 0.1

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def activation(x, w, b):
    """Compute the activation output."""
    return sigmoid(np.dot(x, w) + b)


def update_weights(N, x, y_expected, w, b, eta):
    """Update weights and bias using gradient descent."""
    y = activation(x, w, b)
    gradient = (y - y_expected) * y * (1 - y)

    w -= eta * np.dot(x.T, gradient) / N
    b -= eta * np.sum(gradient) / N
    return w, b


def plot_decision_boundary(ax, x, y, weights, bias, epoch):
    """Plot the data points and the decision boundary."""
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=40, alpha=0.8)
    ax.set_title(f"Epoch {epoch}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    x1_vals = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    x2_vals = -(weights[0] * x1_vals + bias) / weights[1]
    ax.plot(x1_vals, x2_vals, color='black', linewidth=2, label='Decision boundary')
    ax.legend()


def load_data(file_name):
    """Load and preprocess the dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, file_name)
    df = pd.read_csv(path, header=None)
    print("Dataset Iris loaded:")
    print(df.head())

    x = df.iloc[0:100, [0, 1]].values
    y = np.where(df.iloc[0:100, 4].values == 'Iris-setosa', 0, 1)
    return x, y


def split_data(x, y):
    """Split the data into training and testing sets."""
    x_train = np.vstack((x[:40], x[50:90]))
    y_train = np.hstack((y[:40], y[50:90]))
    x_test = np.vstack((x[40:50], x[90:100]))
    y_test = np.hstack((y[40:50], y[90:100]))
    return x_train, y_train, x_test, y_test


def initialize_parameters(features):
    """Initialize weights and bias."""
    weights = np.random.uniform(-0.5, 0.5, size=features)
    bias = np.random.uniform(-0.5, 0.5)
    return weights, bias


def train_model(x_train, y_train, weights, bias, epochs, eta):
    """Train the model and plot decision boundaries at specific epochs."""
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.ravel()
    plot_epochs = np.linspace(0, epochs - 1, 10, dtype=int)

    for epoch in range(epochs):
        weights, bias = update_weights(len(y_train), x_train, y_train, weights, bias, eta)

        if epoch in plot_epochs:
            idx = np.where(plot_epochs == epoch)[0][0]
            plot_decision_boundary(axs[idx], x_train, y_train, weights, bias, epoch)

    plt.tight_layout()
    plt.show()
    return weights, bias


def evaluate_model(x_test, y_test, weights, bias):
    """Evaluate the model on the test set."""
    y_pred_test = activation(x_test, weights, bias)
    accuracy = np.mean((y_pred_test >= 0.5).astype(int) == y_test)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
    return y_pred_test


def visualize_test_predictions(x_test, y_pred_test, weights, bias):
    """Visualize the predictions and decision boundary on the test set."""
    plt.figure(figsize=(8, 6))
    y_pred_labels = (y_pred_test >= 0.5).astype(int)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred_labels, cmap='coolwarm', edgecolor='k', s=40, alpha=0.8)

    x1_vals = np.linspace(np.min(x_test[:, 0]), np.max(x_test[:, 0]), 100)
    x2_vals = -(weights[0] * x1_vals + bias) / weights[1]
    plt.plot(x1_vals, x2_vals, color='black', linewidth=2, label='Decision boundary')

    plt.title("Predictions on test set with decision boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()


def main():
    x, y = load_data(FILE_NAME)
    x_train, y_train, x_test, y_test = split_data(x, y)
    weights, bias = initialize_parameters(FEATURES)

    weights, bias = train_model(x_train, y_train, weights, bias, EPOCHS, ETA)

    y_pred_test = evaluate_model(x_test, y_test, weights, bias)

    visualize_test_predictions(x_test, y_pred_test, weights, bias)


if __name__ == '__main__':
    main()
