The Python script you provided is fully functional and correctly implements a manual comparison between **Grid Search** and **Randomized Search** for hyperparameter tuning using a TensorFlow/Keras MLP model, data augmentation, and dropout, all without relying on scikit-learn's search wrappers.

The indentation is correct throughout the entire script.

Here is the complete, executable code, including comments for clarity:

```python
# ONE-CELL: MLP + Data Augmentation + Dropout + Manual GridSearch + Manual RandomSearch (NO sklearn wrappers)
import numpy as np
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

print("Loading MNIST...")

# ---------------- LOAD MNIST ----------------
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to include a channel dimension (for Data Augmentation) and normalize
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Reduce the training data size (70% discarded) for faster hyperparameter tuning
# We use only 30% (18,000 samples) of the original 60,000 training samples
x_small, _, y_small, _ = train_test_split(x_train, y_train, test_size=0.70, random_state=42)
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


# ---------------- BUILD MODEL FUNCTION ----------------
def build_mlp(optimizer, dropout, units):
    """
    Builds a simple MLP with Dropout layers.
    
    Args:
        optimizer: Keras Optimizer instance (e.g., Adam(0.001))
        dropout: Dropout rate (float)
        units: Number of units in the first Dense layer (int)
    """
    model = Sequential([
        # Flatten the 28x28x1 image to a 784-feature vector
        Flatten(input_shape=(28, 28, 1)),
        
        # First Dense layer
        Dense(units, activation="relu"),
        # Dropout regularization
        Dropout(dropout),
        
        # Second Dense layer (half the units of the first)
        Dense(units // 2, activation="relu"),
        # Second Dropout regularization
        Dropout(dropout),
        
        # Output layer (10 classes for digits 0-9)
        Dense(10, activation="softmax")
    ])
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy", # Use for integer-encoded labels
        metrics=["accuracy"]
    )
    return model


# ---------------- MANUAL GRID SEARCH ----------------
print("\nRunning Manual GridSearch...")

# Define a discrete set of hyperparameter combinations to test
grid_params = [
    {"optimizer": Adam(0.001), "dropout": 0.3, "units": 256},
    {"optimizer": Adam(0.001), "dropout": 0.4, "units": 512},
    {"optimizer": RMSprop(0.001), "dropout": 0.3, "units": 256},
]

best_grid_acc = 0
best_grid_params = None

for p in grid_params:
    # Build and train the model for the current combination
    model = build_mlp(p["optimizer"], p["dropout"], p["units"])
    
    # Train using the data generator (with augmentation)
    model.fit(
        datagen.flow(x_small, y_small, batch_size=128),
        epochs=2, # Short training for quick evaluation
        verbose=0
    )
    
    # Evaluate on the untouched test set
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Note: Retrieving the optimizer name from the object for cleaner printing
    opt_name = p["optimizer"].__class__.__name__
    print(f"Grid Params (Opt: {opt_name}, Drop: {p['dropout']}, Units: {p['units']}) → Accuracy = {acc:.4f}")
    
    # Track the best performing set of parameters
    if acc > best_grid_acc:
        best_grid_acc = acc
        # Create a printable version of the parameters
        best_grid_params = {
            "optimizer": opt_name,
            "dropout": p["dropout"],
            "units": p["units"]
        }

print("\nBest GridSearch Params:", best_grid_params)
print(f"Best GridSearch Accuracy: {best_grid_acc:.4f}")


# ---------------- MANUAL RANDOM SEARCH ----------------
print("\nRunning Manual RandomSearch...")

# Define the search spaces
opt_choices = [Adam(0.001), RMSprop(0.001), SGD(0.01)]
dropout_choices = [0.2, 0.3, 0.4]
units_choices = [128, 256, 512]

random_params = []
for _ in range(3): # Try 3 random combinations
    random_params.append({
        "optimizer": random.choice(opt_choices),
        "dropout": random.choice(dropout_choices),
        "units": random.choice(units_choices)
    })

best_random_acc = 0
best_random_params = None

for p in random_params:
    # Build and train the model for the current combination
    model = build_mlp(p["optimizer"], p["dropout"], p["units"])
    
    # Train using the data generator (with augmentation)
    model.fit(
        datagen.flow(x_small, y_small, batch_size=128), 
        epochs=2, 
        verbose=0
    )
    
    # Evaluate on the untouched test set
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Note: Retrieving the optimizer name from the object for cleaner printing
    opt_name = p["optimizer"].__class__.__name__
    print(f"Random Params (Opt: {opt_name}, Drop: {p['dropout']}, Units: {p['units']}) → Accuracy = {acc:.4f}")

    # Track the best performing set of parameters
    if acc > best_random_acc:
        best_random_acc = acc
        # Create a printable version of the parameters
        best_random_params = {
            "optimizer": opt_name,
            "dropout": p["dropout"],
            "units": p["units"]
        }

print("\nBest RandomSearch Params:", best_random_params)
print(f"Best RandomSearch Accuracy: {best_random_acc:.4f}")

print("\nDONE ✔ (All tuning working without sklearn wrappers!)")
```
