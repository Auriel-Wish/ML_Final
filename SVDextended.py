import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# Load the user attributes
user_attributes = pd.read_csv('data_movie_lens_100k/user_info.csv')  # File with user features
user_attributes.set_index("orig_user_id", inplace=True)

# Load the ratings data
reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5), skip_lines=1)
data = Dataset.load_from_file('data_movie_lens_100k/ratings_all_development_set.csv', reader=reader)

# Convert the data into a Pandas DataFrame for easier merging
ratings_df = pd.read_csv('data_movie_lens_100k/ratings_all_development_set.csv', names=["user_id", "item_id", "rating"], skiprows=1)
ratings_df['user_id'] = ratings_df['user_id'].astype(int)

# Merge user attributes with ratings
ratings_with_features = ratings_df.merge(user_attributes, left_on="user_id", right_index=True)

# Normalize or scale user features (e.g., age and is_male)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

user_features = scaler.fit_transform(user_attributes[["age", "is_male"]])
user_feature_dict = {user_id: features for user_id, features in zip(user_attributes.index, user_features)}

# Define a function to integrate user features into predictions
def hybrid_predict(user_id, item_id, svd_model):
    # Get the latent factors from SVD
    svd_prediction = svd_model.predict(user_id, item_id).est
    
    # Add user feature contribution (e.g., simple linear model for illustration)
    if user_id in user_feature_dict:
        user_feature_vector = user_feature_dict[user_id]
        additional_weight = np.dot(user_feature_vector, np.array([0.1, 0.2]))  # Example weights for age and gender
        return svd_prediction + additional_weight
    return svd_prediction

# Split the data
train_set, test_set = train_test_split(data, test_size=0.2)

# Train SVD and evaluate
for n_factors in [1, 2, 5, 10, 50]:
    # Initialize the model
    model = SVD(n_factors=n_factors)
    
    # Train the model
    model.fit(train_set)
    
    # Predict and incorporate user features
    predictions = [
        hybrid_predict(user_id, item_id, model) for user_id, item_id, _ in test_set
    ]
    
    # Extract true ratings from test set
    true_ratings = [true_rating for _, _, true_rating in test_set]
    
    # Calculate MAE
    mae = np.mean(np.abs(np.array(true_ratings) - np.array(predictions)))
    
    print(f"Number of factors: {n_factors}")
    print(f"Mean Absolute Error (MAE): {mae}")

