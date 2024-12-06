import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from surprise import SVD
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# Initialize the reader
reader = Reader(
    line_format='user item rating', sep=',',
    rating_scale=(1, 5), skip_lines=1
)

# Load the data
data = Dataset.load_from_file(
    'data_movie_lens_100k/ratings_all_development_set.csv', reader=reader
)

# Split the data into train and validation sets
train_set, test_set = train_test_split(data, test_size=0.2)

# Use the SVD algorithm
for n_factors in [1, 2, 5, 10, 50]:
    # Initialize the model with the given number of factors
    model = SVD(n_factors=n_factors)

    # Fit the model on the training set
    model.fit(train_set)

    # Predict on the test set
    predictions = model.test(test_set)

    # Compute Mean Absolute Error (MAE)
    mae = accuracy.mae(predictions)

    print(f"Number of factors: {n_factors}")
    print(f"Mean Absolute Error (MAE): {mae}")
