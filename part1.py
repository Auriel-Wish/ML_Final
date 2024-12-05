# from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem
# from train_valid_test_loader import load_train_valid_test_datasets
from CollabFilterOneVectorPerItem import *
from train_valid_test_loader import *

# Load dataset
train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()

# Define a function to train and evaluate the model
def train_and_evaluate(k, alpha, n_epochs=10, step_size=0.1):
    model = CollabFilterOneVectorPerItem(
        n_epochs=n_epochs, 
        batch_size=10000, 
        step_size=step_size, 
        n_factors=k, 
        alpha=alpha
    )
    model.init_parameter_dict(n_users, n_items, train_tuple)
    model.fit(train_tuple, valid_tuple)

    valid_perf = model.evaluate_perf_metrics(*valid_tuple)
    test_perf = model.evaluate_perf_metrics(*test_tuple)
    
    return model.param_dict, valid_perf, test_perf

# First phase: Train with no regularization and different values of K
results_no_reg = {}
for k in [2, 10, 50]:
    print(f"Training with K={k}, alpha=0.0")
    params, valid_perf, test_perf = train_and_evaluate(k=k, alpha=0.0, n_epochs=20, step_size=0.05)
    results_no_reg[k] = {
        "params": params,
        "valid_perf": valid_perf,
        "test_perf": test_perf,
    }

# # Second phase: Train with moderate regularization for K=50
# print("Training with K=50, moderate regularization (alpha=0.1)")
# params_reg, valid_perf_reg, test_perf_reg = train_and_evaluate(k=50, alpha=0.1, n_epochs=30, step_size=0.05)

# # Save the results for both phases
# results_with_reg = {
#     "params": params_reg,
#     "valid_perf": valid_perf_reg,
#     "test_perf": test_perf_reg,
# }

# # Prepare results for display
# rows = []
# for k, res in results_no_reg.items():
#     rows.append({
#         "K": k,
#         "Alpha": 0.0,
#         "Validation MAE": res["valid_perf"]["mae"],
#         "Test MAE": res["test_perf"]["mae"],
#     })

# rows.append({
#     "K": 50,
#     "Alpha": 0.1,
#     "Validation MAE": results_with_reg["valid_perf"]["mae"],
#     "Test MAE": results_with_reg["test_perf"]["mae"],
# })

# results_df = pd.DataFrame(rows)
# print(results_df)
