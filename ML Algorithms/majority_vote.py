import sys
import numpy as np
from collections import Counter

class MajorityVoteClassifier:
    def __init__(self):
        self.labels = None

    def fit(self, y):
        #Store the labels used for prediction at test time
        self.labels = np.array(y)

    def predict(self, x):
        #Calculate the most common label in the training set
        most_common_label = self.get_most_common_label()

        #Predict the most common label for each point in the dataset using full from
        #https://numpy.org/doc/stable/reference/generated/numpy.full.html
        predictions = np.full(len(x), most_common_label)
        return predictions

    def get_most_common_label(self):
        label_counts = Counter(self.labels)

        #Find the label with the highest count
        max_count = max(label_counts.values())
        #Incase of a tie, get all the candidates with max count and find the value that is numerically
        #higher or comes last alphabetically
        most_common_label_candidates = [label for label, count in label_counts.items() if count == max_count]
        most_common_label = max(most_common_label_candidates)
        return most_common_label

def read_tsv(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        #Skip the first row if it contains column names
        data = [line.strip().split("\t") for line in lines[1:]]
    return data


def write_predictions(predictions, file_path):
    with open(file_path, "w") as file:
        for prediction in predictions:
            file.write(str(prediction) + "\n")

def calculate_error(true_labels, predicted_labels):
    #Find the amount of incorrect predictions
    incorrect_predictions = np.sum(true_labels != predicted_labels)
    total_predictions = len(true_labels)
    #Calculate the error rate
    error = incorrect_predictions / total_predictions
    return error

def main():
    #Parse command-line arguments
    train_input, test_input, train_out, test_out, metrics_out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

    #Load training and test datasets
    train_data = read_tsv(train_input)
    test_data = read_tsv(test_input)

    #Use Try/catch for error handling
    #Extract features and labels, train classifier, make predictions, write metrics and predictions
    try:
        #Extract features (x) and labels (y) for training and test data
        features_train = np.array([list(map(int, row[:-1])) for row in train_data])
        labels_train = np.array([int(row[-1]) for row in train_data])
        features_test = np.array([list(map(int, row[:-1])) for row in test_data])
        labels_test = np.array([int(row[-1]) for row in test_data])

        #Create and train the classifier with labels from training data
        classifier = MajorityVoteClassifier()
        classifier.fit(labels_train)

        #Make predictions on the training and test data with features
        train_predictions = classifier.predict(features_train)
        test_predictions = classifier.predict(features_test)

        #Calculate and write metrics to the metrics_out file
        train_error = calculate_error(labels_train, train_predictions)
        test_error = calculate_error(labels_test, test_predictions)

        #Write errors to metrics_out file
        with open(metrics_out, "w") as metrics_file:
            metrics_file.write(f"error(train): {train_error:.6f}\n")
            metrics_file.write(f"error(test): {test_error:.6f}")
        
        #Write predictions to output files
        write_predictions(train_predictions, train_out)
        write_predictions(test_predictions, test_out)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
