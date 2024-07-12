import numpy as np
import argparse

def sigmoid(x: np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta: np.ndarray,  # shape (D,) where D is feature dim
    X: np.ndarray,      # shape (N, D) where N is num of examples
    y: np.ndarray,      # shape (N,)
    num_epoch: int,
    learning_rate: float
) -> None:
    """
    Trains the logistic regression model using stochastic gradient descent.

    Parameters:
        theta (np.ndarray): Model parameters.
        X (np.ndarray): Input features.
        y (np.ndarray): Target labels.
        num_epoch (int): Number of epochs.
        learning_rate (float): Learning rate for stochastic gradient descent.

    Returns:
        None
    """
    N = X.shape[0]
    for epoch in range(num_epoch):
        for i in range(N):
            #Add bias term
            xi = np.concatenate(([1], X[i]))
            yi = y[i]
            z = np.dot(xi, theta)
            h = sigmoid(z)
            gradient = xi * (h - yi)
            theta -= learning_rate * gradient


def predict(
    theta: np.ndarray,
    X: np.ndarray
) -> np.ndarray:
    """
    Predicts the binary labels using the logistic regression model.

    Parameters:
        theta (np.ndarray): Model parameters.
        X (np.ndarray): Input features.

    Returns:
        np.ndarray: Predicted binary labels.
    """
    #dotproduct with the bias term stacked on.
    z = np.dot(np.hstack((np.ones((X.shape[0], 1)), X)), theta)
    return np.round(sigmoid(z) >= 0.5, decimals=0).astype(int)



def compute_error(
    y_pred: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Computes the classification error.

    Parameters:
        y_pred (np.ndarray): Predicted labels.
        y (np.ndarray): True labels.

    Returns:
        float: Classification error.
    """
    N = y.shape[0]
    correct = np.sum((y_pred >= 0.5) == y)  # Count correct predictions
    error = 1.0 - correct / N
    return error


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int,
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    # Load training data, validation data, test data
    train_data = np.loadtxt(args.train_input)
    validation_data = np.loadtxt(args.validation_input)
    test_data = np.loadtxt(args.test_input)

    # Separate features and labels
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_val, y_val = validation_data[:, 1:], validation_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]

    # Initialize model parameters
    D = X_train.shape[1]  # features
    theta = np.zeros(D + 1)  # Include intercept term

    # Train the model
    train(theta, X_train, y_train, args.num_epoch, args.learning_rate)

    # Predict on training data + test data
    train_predictions = predict(theta, X_train)
    with open(args.train_out, 'w') as f_train_out:
        for prediction in train_predictions:
            f_train_out.write(f"{int(prediction)}\n")
    
    test_predictions = predict(theta, X_test)
    with open(args.test_out, 'w') as f_test_out:
        for prediction in test_predictions:
            f_test_out.write(f"{int(prediction)}\n")


    # Compute metrics
    train_error = compute_error(train_predictions, y_train)
    test_error = compute_error(test_predictions, y_test)

    # Write metrics to file
    with open(args.metrics_out, 'w') as f:
        f.write(f'error(train): {train_error:.6f}\n')
        f.write(f'error(test): {test_error:.6f}\n')
