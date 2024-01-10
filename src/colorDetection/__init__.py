
from pathlib import Path
import numpy as np
import pandas as pd

import csv
import ast

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .utils import pickle_handler
from .utils import logger




class Prep:
    def __init__(self) -> None:
        pass

    def clean_csv(self, input_file, output_file):
        with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Write header to the output file
            writer.writerow(['color_name', 'Red', 'Green', 'Blue'])

            for row in reader:
                color_name, rgb_str = row
                R, G, B = ast.literal_eval(rgb_str)  # Convert the string to a tuple
                writer.writerow((color_name, R, G, B))
            
    def loadDataset(self, dataUrl):
        data = pd.read_csv(dataUrl)    
        X = data.drop('color_name', axis=1)  # X contains all columns except the 'Label' column
        X = X.values.tolist()
        y = data['color_name']  # y contains only the 'Label' column
        return X, y
    
    def rgb_to_embedding(self, rgb_list):
            # Number of possible values for each channel (0-255)
        num_possible_values = 256

        # Create an empty list to store the encoded values
        encoded_values_list = []

        # Iterate through each RGB list in the input
        for rgb_values in rgb_list:
            # Create an empty list to store the one-hot encoded vector
            encoded_vector = []

            # Iterate through each channel (R, G, B)
            for channel_value in rgb_values:
                # Create a binary vector with zeros and set the corresponding index to 1
                binary_vector = [0] * num_possible_values
                binary_vector[channel_value] = 1

                # Extend the encoded vector with the binary vector
                encoded_vector.extend(binary_vector)

            # Append the encoded vector to the list of encoded values
            encoded_values_list.append(encoded_vector)

        return encoded_values_list

# Instantiate the class
dataCleaning = Prep()

class KNNModelHandler:
    def __init__(self, k_neighbors=1):
        self.k_neighbors = k_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.k_neighbors)

    def train_model(self, X_train, y_train):
        # Train the KNN model
        return self.model.fit(X_train, y_train)

    def save_model(self,model, file_path):
        # Save the model to a pickle file
        logger.info(f"saving Model to {file_path}")
        pickle_handler.save_pickle(model, file_path)
        
        

    def load_model(self):
        # Load the model from a pickle file
        pickle_handler.load_latest_pickle()
        

    def predict(self, X):
        # Make predictions using the loaded model
        return self.model.predict(X)


# Instantiate the class
knn_handler = KNNModelHandler()

# # Train the model (provide your actual training data)
# X_train = [[0, 0], [1, 1], [2, 2], [3, 3]]
# y_train = [0, 1, 2, 3]
# knn_handler.train_model(X_train, y_train)

# # Save the model to a pickle file
# knn_handler.save_model('knn_model.pkl')

# # Load the model from the pickle file
# knn_handler.load_model('knn_model.pkl')

# # Make predictions using the loaded model
# X_test = [[4, 4], [5, 5]]
# predictions = knn_handler.predict(X_test)
# print("Predictions:", predictions)


