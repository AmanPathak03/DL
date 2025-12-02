import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from scipy.stats import uniform, randint
# Note: tensorflow.keras is only used here to easily load the MNIST data
from tensorflow.keras.datasets import mnist 

print("Loading datasets...")

# ------------------- CREDIT APPROVAL DATASET PREPARATION -------------------
print("Preparing Credit Approval data...")
credit = fetch_openml("credit-approval", version=1, as_frame=True)
Xc = credit.data.copy()
yc = credit.target.copy()

# The Credit Approval dataset has mixed types and '?' missing values.
# Convert ALL columns to string first (critical fix) to handle '?' and then Label Encode.
Xc = Xc.applymap(str)

# Label encode each categorical column separately
for col in Xc.columns:
    le = LabelEncoder()
    Xc[col] = le.fit_transform(Xc[col])

# Encode target y
yc = LabelEncoder().fit_transform(yc)

# Train-test split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42
)

# Scale features
scaler_c = StandardScaler()
Xc_train = scaler_c.fit_transform(Xc_train)
Xc_test = scaler_c.transform(Xc_test)
print(f"Credit Train shape: {Xc_train.shape}, Test shape: {Xc_test.shape}")


# ------------------- MNIST DATASET PREPARATION -------------------
print("Preparing MNIST data...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten 28x28 images to 784 features and normalize pixel values (0-255 to 0.0-1.0)
X_m_train = x_train.reshape(-1, 784) / 255.0
X_m_test = x_test.reshape(-1, 784) / 255.0
print(f"MNIST Train shape: {X_m_train.shape}, Test shape: {X_m_test.shape}")


# ------------------- HYPERPARAMETER SEARCH SPACES -------------------
# Grid Search space: discrete combinations
param_grid = {
    "hidden_layer_sizes": [(50,), (100,), (100, 50)], # 3 options
    "activation": ["relu", "tanh"],                   # 2 options
    "learning_rate_init": [0.001, 0.01],              # 2 options
    # Total GridSearch runs per CV fold: 3 * 2 * 2 = 12
}

# Random Search space: distribution-based
param_random = {
    # Single layer with units between 50 and 150 (inclusive of 50, exclusive of 150)
    "hidden_layer_sizes": [ (randint.rvs(50, 150),) ], 
    "activation": ["relu", "tanh"],
    # Learning rate between 0.0005 and (0.0005 + 0.02) = 0.0205
    "learning_rate_init": uniform(0.0005, 0.02), 
}


# ------------------- CREDIT APPROVAL GRIDSEARCH -------------------
print("\n--- 1. Credit Approval: GridSearch ---")
mlp_c = MLPClassifier(max_iter=200, random_state=42)
# CV=3 means 3-fold cross-validation. Total fits: 12 * 3 = 36
grid_c = GridSearchCV(mlp_c, param_grid, cv=3, n_jobs=-1)
grid_c.fit(Xc_train, yc_train)
pred_c = grid_c.predict(Xc_test)
acc_c = accuracy_score(yc_test, pred_c)
print("Best params (GridSearch - Credit):", grid_c.best_params_)
print(f"Test Accuracy: {acc_c:.4f}")

# ------------------- CREDIT APPROVAL RANDOM SEARCH -------------------
print("\n--- 2. Credit Approval: RandomSearch ---")
mlp_c2 = MLPClassifier(max_iter=200, random_state=42)
# n_iter=5 means 5 random combinations are tested. Total fits: 5 * 3 = 15
rand_c = RandomizedSearchCV(mlp_c2, param_random, n_iter=5, cv=3, n_jobs=-1, random_state=42)
rand_c.fit(Xc_train, yc_train)
pred_c2 = rand_c.predict(Xc_test)
acc_c2 = accuracy_score(yc_test, pred_c2)
print("Best params (RandomSearch - Credit):", rand_c.best_params_)
print(f"Test Accuracy: {acc_c2:.4f}")


# ------------------- MNIST GRIDSEARCH -------------------
print("\n--- 3. MNIST: GridSearch (10k sample) ---")
# Use a smaller sample of MNIST due to computational cost
idx = np.random.choice(len(X_m_train), 10000, replace=False)
X_small = X_m_train[idx]
y_small = y_train[idx]

# max_iter is reduced to 20 for faster initial comparison
mlp_m = MLPClassifier(max_iter=20, random_state=42) 
# CV=2 means 2-fold cross-validation. Total fits: 12 * 2 = 24
grid_m = GridSearchCV(mlp_m, param_grid, cv=2, n_jobs=-1)
grid_m.fit(X_small, y_small)

# Evaluate on a sample of the test set
pred_m = grid_m.predict(X_m_test[:5000])
acc_m = accuracy_score(y_test[:5000], pred_m)

print("Best params (GridSearch - MNIST):", grid_m.best_params_)
print(f"Test Accuracy: {acc_m:.4f}")

# ------------------- MNIST RANDOM SEARCH -------------------
print("\n--- 4. MNIST: RandomSearch (10k sample) ---")
# n_iter=5 means 5 random combinations are tested. Total fits: 5 * 2 = 10
rand_m = RandomizedSearchCV(
    MLPClassifier(max_iter=20, random_state=42),
    param_random, n_iter=5, cv=2, n_jobs=-1, random_state=42
)
rand_m.fit(X_small, y_small)

# Evaluate on a sample of the test set
pred_m2 = rand_m.predict(X_m_test[:5000])
acc_m2 = accuracy_score(y_test[:5000], pred_m2)

print("Best params (RandomSearch - MNIST):", rand_m.best_params_)
print(f"Test Accuracy: {acc_m2:.4f}")

print("\n\nDONE ✔ (Hyperparameter tuning comparison complete)")
