import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

# Load the user attributes
user_attributes = pd.read_csv('data_movie_lens_100k/user_info.csv')  # File with user features
user_attributes.set_index("orig_user_id", inplace=True)

# Load movie attributes
movie_attributes = pd.read_csv('data_movie_lens_100k/movie_info.csv', names=["item_id", "title", "release_year"], skiprows=1)
movie_attributes.set_index("item_id", inplace=True)

# Load the ratings data
reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5), skip_lines=1)
data = Dataset.load_from_file('data_movie_lens_100k/ratings_all_development_set.csv', reader=reader)

# Convert the data into a Pandas DataFrame for easier merging
ratings_df = pd.read_csv('data_movie_lens_100k/ratings_all_development_set.csv', names=["user_id", "item_id", "rating"], skiprows=1)
ratings_df['user_id'] = ratings_df['user_id'].astype(int)

# Merge user attributes with ratings
ratings_with_features = ratings_df.merge(user_attributes, left_on="user_id", right_index=True)

# Normalize or scale user features (e.g., age and is_male)
scaler = StandardScaler()
user_features = scaler.fit_transform(user_attributes[["age", "is_male"]])
user_feature_dict = {user_id: features for user_id, features in zip(user_attributes.index, user_features)}

# Normalize or scale movie features (e.g., release_year)
movie_features = scaler.fit_transform(movie_attributes[["release_year"]])
movie_feature_dict = {movie_id: features for movie_id, features in zip(movie_attributes.index, movie_features)}

# Define a function to integrate user and movie features into predictions
def hybrid_predict(user_id, item_id, svd_model, user_weights, movie_weights, l1_reg):
    # Get the latent factors from SVD
    svd_prediction = svd_model.predict(user_id, item_id).est
    
    # Add user feature contribution (e.g., simple linear model for illustration)
    additional_weight = 0
    if user_id in user_feature_dict:
        user_feature_vector = user_feature_dict[user_id]
        additional_weight += np.dot(user_feature_vector, user_weights) - l1_reg * np.sum(np.abs(user_weights))
    
    # Add movie feature contribution (e.g., release year)
    if item_id in movie_feature_dict:
        movie_feature_vector = movie_feature_dict[item_id]
        additional_weight += np.dot(movie_feature_vector, movie_weights) - l1_reg * np.sum(np.abs(movie_weights))
    
    return svd_prediction + additional_weight

# Define the parameter grid for grid search
#current best {'l1_reg': 0.1, 'learning_rate': 0.02, 'movie_weights': [0.05], 'n_factors': 2, 'user_weights': [0.2, 0.3]}
# Best Mean Absolute Error (MAE): 0.7241092812788638
param_grid = {
    'n_factors': [2, 10, 50],
    'user_weights': [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
    'movie_weights': [[0.01], [0.05], [0.1]],
    'l1_reg': [0.05, 0.1, 0.7],
    'learning_rate': [0.02, 0.03, 0.04],
    'epochs': [10, 20, 30]
}

# Split the data
train_set, test_set = train_test_split(data, test_size=0.2)

# Perform grid search
best_mae = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    print(f"Training with parameters: {params}")
    
    # Initialize the model with the current learning rate and number of epochs
    model = SVD(n_factors=params['n_factors'], lr_all=params['learning_rate'], n_epochs=params['epochs'])
    
    # Train the model
    model.fit(train_set)
    
    # Predict and incorporate user and movie features
    predictions = []
    for user_id, item_id, _ in test_set:
        prediction = hybrid_predict(user_id, item_id, model, params['user_weights'], params['movie_weights'], params['l1_reg'])
        predictions.append(prediction)
    
    # Extract true ratings from test set
    true_ratings = [true_rating for _, _, true_rating in test_set]
    
    # Calculate MAE
    mae = np.mean(np.abs(np.array(true_ratings) - np.array(predictions)))
    print(f"MAE for current parameters: {mae}")
    
    if mae < best_mae:
        best_mae = mae
        best_params = params

print(f"Best parameters: {best_params}")
print(f"Best Mean Absolute Error (MAE): {best_mae}")
