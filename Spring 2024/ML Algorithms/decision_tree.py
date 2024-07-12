import numpy as np
import sys

class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None
        self.feature_names = None

    def train(self, X, y, feature_names):
        self.tree = self.build_tree(X, y, depth=0)
        self.feature_names = feature_names
        
    def calculate_entropy(self, labels):
        p0 = np.mean(labels == 0)
        p1 = np.mean(labels == 1)
        entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1)) if p0 != 0 and p1 != 0 else 0
        return entropy
    
    def calculate_mutual_information(self, feature, labels):
        # Calculate the entropy of the initial label
        mutual_info = self.calculate_entropy(labels)
        
        # Get unique feature values
        unique_values = np.unique(feature)
        
        for val in unique_values:
            # Get subset of labels corresponding to the current feature value
            subset_labels = labels[feature == val]
            
            # Calculate probability of the current feature value
            probability = len(subset_labels) / len(labels)
            
            # Update mutual information
            mutual_info -= probability * self.calculate_entropy(subset_labels)
        
        return mutual_info
    
    def build_tree(self, X, y, depth):
        
        # Base case: if maximum depth is reached or there's only one value in labels
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {'label_distribution': [np.sum(y == 0), np.sum(y == 1)]}
    
        # Calculate mutual information for each feature
        mutual_infos = []
        for feature_index in range(len(X[0])):
            mutual_info = self.calculate_mutual_information(X[:, feature_index], y)
            mutual_infos.append((feature_index, mutual_info))
    
        # Select the feature with the highest mutual information using lambda function
        # Lambda function just takes in the mutual_infos list and returns the vals in the key-val pair.
        best_feature_index, best_mutual_info = max(mutual_infos, key=lambda x: x[1])
    
        # If no mutual information or less than 0, return leaf node
        if best_mutual_info <= 0:
            return {'label_distribution': [np.sum(y == 0), np.sum(y == 1)]}
    
        # Get unique values of the best feature and assign it as the current node. Incase of a tie, only take the first value
        best_feature_values = np.unique(X[:, best_feature_index])
        best_feature_value = best_feature_values[0]
    
        # Split the dataset into left and right branches
        # left side creates first, right is the other feature value
        left_indices = X[:, best_feature_index] == best_feature_value
        right_indices = ~left_indices
    
        left_node = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self.build_tree(X[right_indices], y[right_indices], depth + 1)
    
        return {
            'label_distribution': [np.sum(y == 0), np.sum(y == 1)],
            'index': best_feature_index,
            'value': best_feature_value,
            'left': left_node,
            'right': right_node
        }


    def predict(self, X):
        predictions = []
        for feature in X:
            node = self.tree
            while 'left' in node and 'right' in node:
                feature_value = feature[node['index']]
                # If the feature value is 0, then we can assume that the node is left.
                if feature_value == 0: 
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(1 if node['label_distribution'][1] > node['label_distribution'][0] else 0)
        return predictions


    def print_tree(self, node, depth, prefix="", print_list=None):
        # Initialize the print_list if it's not provided
        if print_list is None:
            print_list = []
        
        if 'label_distribution' in node:
            print_list.append("| " * depth + f"{prefix}[{node['label_distribution'][0]} 0/{node['label_distribution'][1]} 1]")
        if 'index' in node:
            feature_name = self.feature_names[node['index']]
            if 'left' in node:
                self.print_tree(node['left'], depth + 1, prefix=f"{feature_name} = 0: ", print_list=print_list)
            if 'right' in node:
                self.print_tree(node['right'], depth + 1, prefix=f"{feature_name} = 1: ", print_list=print_list)
        
        return print_list



def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Read and save feature names from the first row for printing purposes
        feature_names = lines[0].strip().split('\t')  
        data = np.loadtxt(lines[1:], delimiter='\t')
        X = data[:, :-1]
        y = data[:, -1]
    return X, y, feature_names

def calculate_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

def write_metrics(train_error, test_error, output_file):
    with open(output_file, 'w') as f:
        f.write(f"error(train): {train_error:.6f}\n")
        f.write(f"error(test): {test_error:.6f}\n")

def write_labels(labels, output_file):
    with open(output_file, 'w') as f:
        for label in labels:
            f.write(f"{int(label)}\n")

def main():
    if len(sys.argv) != 8:
        print("Usage: python decision_tree.py <train input> <test input> <max depth> <train out> <test out> <metrics out> <print out>")
        sys.exit(1)

    train_input, test_input, max_depth, train_out, test_out, metrics_out, print_out = sys.argv[1:]

    max_depth = int(max_depth)
    initial_depth = 0
    
    X_train, y_train, feature_names = load_data(train_input)
    X_test, y_test, _ = load_data(test_input)

    #Train decision tree classifier
    dt = DecisionTree(max_depth=max_depth)
    dt.train(X_train, y_train, feature_names)
    print_list = dt.print_tree(dt.tree, depth=initial_depth)
    # Output list to print_out
    with open(print_out, "w") as file:
        for line in print_list:
            file.write(line + "\n")
            
    #Make predictions on training and test data based on trained classifier
    train_predictions = dt.predict(X_train)
    test_predictions = dt.predict(X_test)

    write_labels(train_predictions, train_out)
    write_labels(test_predictions, test_out)

    train_error = calculate_error(y_train, train_predictions)
    test_error = calculate_error(y_test, test_predictions)

    write_metrics(train_error, test_error, metrics_out)


if __name__ == "__main__":
    main()
