import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam

print("TF version:", tf.__version__)

# ---------------------- DATASET ----------------------
## Generate and split the "moons" data
X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

## Standardize the features (important for NN training)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------------------- MODEL BUILDER ----------------------
def build_mlp(act, lr=0.005):
    """
    Builds a simple two-layer MLP with a specified activation function.
    """
    model = Sequential()
    
    # Handle LeakyReLU separately as it's a specific layer
    if act == "leaky":
        model.add(Dense(32, input_shape=(2,)))
        model.add(LeakyReLU(negative_slope=0.1))
        model.add(Dense(32))
        model.add(LeakyReLU(negative_slope=0.1))
    # Use standard activation for others
    else:
        model.add(Dense(32, activation=act, input_shape=(2,)))
        model.add(Dense(32, activation=act))
        
    # Output layer for binary classification
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ---------------------- TRAIN & COMPARE ----------------------
activations = ["sigmoid", "tanh", "relu", "elu", "leaky"]
results = {}

for act in activations:
    print(f"\nTraining activation: {act}")
    
    # Use different learning rates based on common practice for these functions
    lr = 0.01 if act in ["sigmoid", "tanh"] else 0.005
    
    model = build_mlp(act, lr=lr)
    
    # Early Stopping to prevent overfitting and speed up training
    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=120, batch_size=32,
        validation_split=0.15,
        verbose=0, callbacks=[cb]
    )
    
    # Evaluate model performance on the test set
    preds = (model.predict(X_test, verbose=0) > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    results[act] = acc
    
    print(f" -> Test Accuracy: {acc:.4f} | Epochs trained: {len(history.history['loss'])}")

# ---------------------- PLOT TRAINING ACCURACIES ----------------------
plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values(), color=['skyblue', 'salmon', 'lightgreen', 'gold', 'mediumpurple'])
plt.title("Activation Function Performance on Moons Dataset")
plt.ylabel("Test Accuracy")
plt.xlabel("Activation Function")
plt.ylim(min(results.values())*0.9, 1.0)
plt.grid(axis='y', linestyle='--')
plt.show() # 

# ---------------------- PRINT SUMMARY ----------------------
print("\nSummary of test accuracies:")
for act, acc in results.items():
    print(f"{act:7s} : {acc:.4f}")
