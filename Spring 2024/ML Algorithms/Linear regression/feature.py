import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################

def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset

def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def trim(review, glove_map):
    """
    Trims a review by removing out-of-vocabulary words and calculating
    the GloVe feature vectors for the remaining words.

    Parameters:
        review: The input review.
        glove_map: A dictionary containing GloVe embeddings.

    Returns:
        np.ndarray: The feature vector.
    """
    words = review.strip().split()
    feature_vectors = []
    for word in words:
        if word in glove_map:
            feature_vectors.append(glove_map[word])
            
    if len(feature_vectors) > 0:
        feature_vectors = np.array(feature_vectors)
        return np.mean(feature_vectors, axis=0).round(decimals=6)
    else:
        # If all words in the review are out-of-vocabulary, return a vector of zeros
        return np.zeros(VECTOR_LEN).round(decimals=6)

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

    # Load GloVe feature dictionary + datasets
    glove_map = load_feature_dictionary(args.feature_dictionary_in)
    train_data = load_tsv_dataset(args.train_input)
    validation_data = load_tsv_dataset(args.validation_input)
    test_data = load_tsv_dataset(args.test_input)

    # Write to the training out file
    with open(args.train_out, 'w') as f_train:
        for label, review in train_data:
            feature_vector = trim(review, glove_map)
            # Write label + feature to file
            f_train.write(f'{label:.6f}\t')
            for num in feature_vector:
                f_train.write(f'{num:.6f}\t')
    
            f_train.write('\n')

    # Write to the validation out file
    with open(args.validation_out, 'w') as f_validation:
        for label, review in validation_data:
            feature_vector = trim(review, glove_map)
            # Write label + feature to file
            f_validation.write(f'{label:.6f}\t')
    
            for num in feature_vector:
                f_validation.write(f'{num:.6f}\t')
    
            f_validation.write('\n')

    # Write to the test out file
    with open(args.test_out, 'w') as f_test:
        for label, review in test_data:
            feature_vector = trim(review, glove_map)
            # Write label + feature to file
            f_test.write(f'{label:.6f}\t')
            for num in feature_vector:
                f_test.write(f'{num:.6f}\t')
    
            f_test.write('\n')