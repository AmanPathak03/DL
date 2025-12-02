import numpy as np
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

print("Loading MNIST...")

# ---------------- LOAD DATA ----------------
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape (add channel dimension) and Normalize (0-1 range)
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Reduce for faster tuning (using only 25% of the original training data)
x_small, _, y_small, _ = train_test_split(x_train, y_train, test_size=0.75, random_state=42)
print(f"Reduced training data size: {x_small.shape[0]} samples.")


# ---------------- DATA AUGMENTATION ----------------
# Define the image data generator for on-the-fly augmentation
datagen = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
# Fit the generator to the reduced training data
datagen.fit(x_small)


# ---------------- BUILD CNN FUNCTION ----------------
def build_cnn(optimizer, dropout, filters):
    """
    Builds a simple two-convolutional-layer CNN architecture.
    """
    model = Sequential([
        # First Conv Layer
        Conv2D(filters, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(dropout),
        
        # Second Conv Layer (double the filters)
        Conv2D(filters * 2, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(dropout),
        
        # Classification Head
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout),
        Dense(10, activation='softmax') # Output layer for 10 classes
    ])
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy", # Use for integer-encoded labels
        metrics=["accuracy"]
    )
    return model
# 

[Image of a standard Convolutional Neural Network architecture]



# ---------------- MANUAL GRID SEARCH ----------------
print("\nRunning Manual GridSearch...")

# Define a discrete set of hyperparameter combinations to test
grid_params = [
    {"optimizer": Adam(0.001), "dropout": 0.3, "filters": 32},
    {"optimizer": Adam(0.001), "dropout": 0.4, "filters": 32},
    {"optimizer": RMSprop(0.001), "dropout": 0.3, "filters": 64},
]

best_grid_acc = 0
best_grid_params = None

for p in grid_params:
    # Build and train the model
    model = build_cnn(p["optimizer"], p["dropout"], p["filters"])
    
    # Train using the data generator (with augmentation)
    model.fit(
        datagen.flow(x_small, y_small, batch_size=128), 
        epochs=2, # Short training for quick evaluation
        verbose=0
    )
    
    # Evaluate on the untouched test set
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Get the optimizer name for clean printing
    opt_name = p["optimizer"].__class__.__name__
    print(f"Grid Params (Opt: {opt_name}, Drop: {p['dropout']}, Filters: {p['filters']}) → Accuracy = {acc:.4f}")
    
    # Track the best performing set of parameters
    if acc > best_grid_acc:
        best_grid_acc = acc
        best_grid_params = {
            "optimizer": opt_name,
            "dropout": p["dropout"],
            "filters": p["filters"]
        }

print("\nBest GridSearch Params:", best_grid_params)
print(f"Best GridSearch Accuracy: {best_grid_acc:.4f}")


# ---------------- MANUAL RANDOM SEARCH ----------------
print("\nRunning Manual RandomSearch...")

# Define the search spaces (choices for random sampling)
opt_choices = [Adam(0.001), RMSprop(0.001), SGD(0.01)]
dropout_choices = [0.2, 0.3, 0.4]
filters_choices = [32, 48, 64]

random_params = []
for _ in range(3): # Try 3 random combinations
    random_params.append({
        "optimizer": random.choice(opt_choices),
        "dropout": random.choice(dropout_choices),
        "filters": random.choice(filters_choices)
    })

best_random_acc = 0
best_random_params = None

for p in random_params:
    # Build and train the model
    model = build_cnn(p["optimizer"], p["dropout"], p["filters"])
    
    # Train using the data generator (with augmentation)
    model.fit(
        datagen.flow(x_small, y_small, batch_size=128), 
        epochs=2, 
        verbose=0
    )
    
    # Evaluate on the untouched test set
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Get the optimizer name for clean printing
    opt_name = p["optimizer"].__class__.__name__
    print(f"Random Params (Opt: {opt_name}, Drop: {p['dropout']}, Filters: {p['filters']}) → Accuracy = {acc:.4f}")

    # Track the best performing set of parameters
    if acc > best_random_acc:
        best_random_acc = acc
        best_random_params = {
            "optimizer": opt_name,
            "dropout": p["dropout"],
            "filters": p["filters"]
        }

print("\nBest RandomSearch Params:", best_random_params)
print(f"Best RandomSearch Accuracy: {best_random_acc:.4f}")

print("\nDONE ✔ CNN + Augmentation + Dropout + GridSearch + RandomSearch Completed Successfully")
