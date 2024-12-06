import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV

# Load the training data
reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5), skip_lines=1)
data = Dataset.load_from_file('data_movie_lens_100k/ratings_all_development_set.csv', reader=reader)

# Split the data for grid search
train_set, test_set = data.build_full_trainset(), None

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_factors': [20, 50, 100],
    'n_epochs': [10, 20, 30],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.05, 0.1]
}

# Perform grid search
grid_search = GridSearchCV(
    SVD,
    param_grid,
    measures=['rmse'],
    cv=3,
    n_jobs=-1
)
grid_search.fit(data)

# Get the best parameters
best_params = grid_search.best_params['rmse']
print("Best Parameters:", best_params)

# Retrain the model on the full dataset
full_train_set = data.build_full_trainset()
best_model = SVD(
    n_factors=best_params['n_factors'],
    n_epochs=best_params['n_epochs'],
    lr_all=best_params['lr_all'],
    reg_all=best_params['reg_all']
)
best_model.fit(full_train_set)

# Convert Surprise internal mappings to strings
train_users = {str(uid): inner_id for uid, inner_id in best_model.trainset._raw2inner_id_users.items()}
train_items = {str(iid): inner_id for iid, inner_id in best_model.trainset._raw2inner_id_items.items()}

# Load the leaderboard data (final testing data)
leaderboard_df = pd.read_csv('data_movie_lens_100k/ratings_masked_leaderboard_set.csv')

# Ensure IDs in the leaderboard are strings
leaderboard_df['user_id'] = leaderboard_df['user_id'].astype(str).str.strip()
leaderboard_df['item_id'] = leaderboard_df['item_id'].astype(str).str.strip()

# Check for commonality after conversion
common_users = set(leaderboard_df['user_id']) & set(train_users.keys())
common_items = set(leaderboard_df['item_id']) & set(train_items.keys())

print(f"Common Users: {len(common_users)}")
print(f"Common Items: {len(common_items)}")

# Generate predictions for the leaderboard set
predictions = []
for _, row in leaderboard_df.iterrows():
    user_id = row['user_id']
    item_id = row['item_id']
    
    # Check if the user or item is unseen and handle appropriately
    if user_id in train_users and item_id in train_items:
        # Predict the rating for the user-item pair
        predicted_rating = best_model.predict(user_id, item_id).est
    else:
        # Use global mean if the user or item is unseen
        predicted_rating = best_model.trainset.global_mean
    
    predictions.append(predicted_rating)

# Save predictions to a plain text file
predictions = pd.Series(predictions)  # Convert to a pandas Series for saving
predictions.to_csv('predicted_ratings_leaderboard.txt', index=False, header=False)

print("Predictions saved to predicted_ratings_leaderboard.txt")