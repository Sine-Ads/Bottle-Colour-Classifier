import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
 

 
IMG_SIZE = 64  # resize all images to 64x64
FOLDERS = ['Dark_blue', 'Dark_green_blue', 'Orange', 'Pink'] 
 
X_list, Y_list = [], []
 
for label, folder in enumerate(FOLDERS):
    folder_path = f'Data_set for task 1/{folder}'
    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith('.jpg') or img_file.lower().endswith('.jpeg'):
            img = Image.open(f'{folder_path}/{img_file}').convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            X_list.append(np.array(img).flatten())
            Y_list.append(label)
 
X = np.array(X_list).T / 255.0   # shape: (12288, 60)
Y = np.array(Y_list)              # shape: (60,)
 
# Shuffle
np.random.seed(42)
indices = np.random.permutation(X.shape[1])
X, Y = X[:, indices], Y[indices]
 
# Train / dev split (50 train, 10 dev)
X_dev,   Y_dev   = X[:, :10],  Y[:10]
X_train, Y_train = X[:, 10:],  Y[10:]
 
print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"X_dev:   {X_dev.shape},   Y_dev:   {Y_dev.shape}")
 

INPUT_SIZE  = IMG_SIZE * IMG_SIZE * 3   # 12288
HIDDEN_SIZE = 16
NUM_CLASSES = 4
 
def init_params():
    W1 = np.random.randn(HIDDEN_SIZE, INPUT_SIZE) * 0.01
    b1 = np.zeros((HIDDEN_SIZE, 1))
    W2 = np.random.randn(NUM_CLASSES, HIDDEN_SIZE) * 0.01
    b2 = np.zeros((NUM_CLASSES, 1))
    return W1, b1, W2, b2
 
def ReLU(Z):
    return np.maximum(0, Z)
 
def softmax(Z):
    Z -= np.max(Z, axis=0)
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)
 
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
 
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, NUM_CLASSES))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T
 
def deriv_ReLU(Z):
    return Z > 0
 
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2
 
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2
 
def get_predictions(A2):
    return np.argmax(A2, 0)
 
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size
 
def compute_loss(A2, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    loss = -np.sum(one_hot_Y * np.log(A2 + 1e-8)) / m
    return loss

# 3. TRAINING WITH EXPERIMENT TRACKING

def train(X_train, Y_train, X_dev, Y_dev, iterations=500, alpha=0.1, label='experiment'):
    W1, b1, W2, b2 = init_params()
 
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
 
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X_train, Y_train)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
 
        if i % 50 == 0:
            # Train metrics
            train_loss = compute_loss(A2, Y_train)
            train_acc  = get_accuracy(get_predictions(A2), Y_train)
 
            # Val metrics
            _, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
            val_loss = compute_loss(A2_dev, Y_dev)
            val_acc  = get_accuracy(get_predictions(A2_dev), Y_dev)
 
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
 
            print(f"[{label}] Iter {i:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
 
    return train_losses, val_losses, train_accs, val_accs
 

# 4. RUN EXPERIMENTS

iterations  = 500
x_axis      = list(range(0, iterations, 50))
results     = {}
 
# Experiment 1: Different learning rates
print("\n========== EXPERIMENT 1: Learning Rates ==========")
for lr in [0.001, 0.01, 0.1, 0.5]:
    tl, vl, ta, va = train(X_train, Y_train, X_dev, Y_dev, iterations, alpha=lr, label=f'lr={lr}')
    results[f'lr={lr}'] = (tl, vl, ta, va)
 

# 5. PLOT RESULTS

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Experiment 1: Effect of Learning Rate', fontsize=14)
 
for key, (tl, vl, ta, va) in results.items():
    axes[0].plot(x_axis, tl, label=f'{key} train')
    axes[0].plot(x_axis, vl, linestyle='--', label=f'{key} val')
    axes[1].plot(x_axis, ta, label=f'{key} train')
    axes[1].plot(x_axis, va, linestyle='--', label=f'{key} val')
 
axes[0].set_title('Loss'); axes[0].set_xlabel('Iteration'); axes[0].legend(fontsize=7)
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Iteration'); axes[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig('experiment1_learning_rates.png', dpi=150)
plt.show()
print("Saved: experiment1_learning_rates.png")
 

# EXPERIMENT 2: Batch Size

def train_minibatch(X_train, Y_train, X_dev, Y_dev, iterations=500, alpha=0.01, batch_size=50, label='experiment'):
    W1, b1, W2, b2 = init_params()
    m = X_train.shape[1]
 
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
 
    for i in range(iterations):
        # Shuffle training data each iteration
        perm = np.random.permutation(m)
        X_shuffled = X_train[:, perm]
        Y_shuffled = Y_train[perm]
 
        # Mini-batch loop
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[start:end]
 
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X_batch, Y_batch)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
 
        if i % 50 == 0:
            # Evaluate on full train set
            _, _, _, A2_train = forward_prop(W1, b1, W2, b2, X_train)
            train_loss = compute_loss(A2_train, Y_train)
            train_acc  = get_accuracy(get_predictions(A2_train), Y_train)
 
            # Evaluate on val set
            _, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
            val_loss = compute_loss(A2_dev, Y_dev)
            val_acc  = get_accuracy(get_predictions(A2_dev), Y_dev)
 
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
 
            print(f"[{label}] Iter {i:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
 
    return train_losses, val_losses, train_accs, val_accs
 
print("\n========== EXPERIMENT 2: Batch Size ==========")
results2 = {}
for bs in [5, 10, 25, 50]:  # 50 = full batch (same as experiment 1)
    tl, vl, ta, va = train_minibatch(X_train, Y_train, X_dev, Y_dev, iterations=500, alpha=0.01, batch_size=bs, label=f'batch={bs}')
    results2[f'batch={bs}'] = (tl, vl, ta, va)
 
# Plot Experiment 2
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Experiment 2: Effect of Batch Size', fontsize=14)
 
for key, (tl, vl, ta, va) in results2.items():
    axes[0].plot(x_axis, tl, label=f'{key} train')
    axes[0].plot(x_axis, vl, linestyle='--', label=f'{key} val')
    axes[1].plot(x_axis, ta, label=f'{key} train')
    axes[1].plot(x_axis, va, linestyle='--', label=f'{key} val')
 
axes[0].set_title('Loss'); axes[0].set_xlabel('Iteration'); axes[0].legend(fontsize=7)
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Iteration'); axes[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig('experiment2_batch_size.png', dpi=150)
plt.show()
print("Saved: experiment2_batch_size.png")


# EXPERIMENT 3: Dropout

def forward_prop_dropout(W1, b1, W2, b2, X, dropout_rate=0.0, training=True):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    
    # Apply dropout only during training
    if training and dropout_rate > 0:
        mask = (np.random.rand(*A1.shape) > dropout_rate)
        A1 = A1 * mask / (1 - dropout_rate)  # scale to keep expected value same
    
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def train_dropout(X_train, Y_train, X_dev, Y_dev, iterations=500, alpha=0.01, dropout_rate=0.0, label='experiment'):
    W1, b1, W2, b2 = init_params()

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for i in range(iterations):
        # Training with dropout
        Z1, A1, Z2, A2 = forward_prop_dropout(W1, b1, W2, b2, X_train, dropout_rate=dropout_rate, training=True)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X_train, Y_train)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 50 == 0:
            # Evaluate WITHOUT dropout
            _, _, _, A2_train = forward_prop_dropout(W1, b1, W2, b2, X_train, dropout_rate=0.0, training=False)
            train_loss = compute_loss(A2_train, Y_train)
            train_acc  = get_accuracy(get_predictions(A2_train), Y_train)

            _, _, _, A2_dev = forward_prop_dropout(W1, b1, W2, b2, X_dev, dropout_rate=0.0, training=False)
            val_loss = compute_loss(A2_dev, Y_dev)
            val_acc  = get_accuracy(get_predictions(A2_dev), Y_dev)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f"[{label}] Iter {i:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

    return train_losses, val_losses, train_accs, val_accs

print("\n========== EXPERIMENT 3: Dropout ==========")
results3 = {}
x_axis3 = list(range(0, 500, 50))
for dr in [0.0, 0.2, 0.5, 0.8]:
    tl, vl, ta, va = train_dropout(X_train, Y_train, X_dev, Y_dev, iterations=500, alpha=0.01, dropout_rate=dr, label=f'dropout={dr}')
    results3[f'dropout={dr}'] = (tl, vl, ta, va)

# Plot Experiment 3
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Experiment 3: Effect of Dropout', fontsize=14)

for key, (tl, vl, ta, va) in results3.items():
    axes[0].plot(x_axis3, tl, label=f'{key} train')
    axes[0].plot(x_axis3, vl, linestyle='--', label=f'{key} val')
    axes[1].plot(x_axis3, ta, label=f'{key} train')
    axes[1].plot(x_axis3, va, linestyle='--', label=f'{key} val')

axes[0].set_title('Loss'); axes[0].set_xlabel('Iteration'); axes[0].legend(fontsize=7)
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Iteration'); axes[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig('experiment3_dropout.png', dpi=150)
plt.show()
print("Saved: experiment3_dropout.png")


# EXPERIMENT 4: L2 Regularization

def back_prop_l2(Z1, A1, Z2, A2, W2, W1, X, Y, lambda_reg=0.0):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T) + (lambda_reg/m) * W2  # L2 term
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T) + (lambda_reg/m) * W1  # L2 term
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def train_l2(X_train, Y_train, X_dev, Y_dev, iterations=500, alpha=0.01, lambda_reg=0.0, label='experiment'):
    W1, b1, W2, b2 = init_params()

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
        dW1, db1, dW2, db2 = back_prop_l2(Z1, A1, Z2, A2, W2, W1, X_train, Y_train, lambda_reg)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 50 == 0:
            _, _, _, A2_train = forward_prop(W1, b1, W2, b2, X_train)
            train_loss = compute_loss(A2_train, Y_train)
            train_acc  = get_accuracy(get_predictions(A2_train), Y_train)

            _, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
            val_loss = compute_loss(A2_dev, Y_dev)
            val_acc  = get_accuracy(get_predictions(A2_dev), Y_dev)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f"[{label}] Iter {i:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

    return train_losses, val_losses, train_accs, val_accs

print("\n========== EXPERIMENT 4: L2 Regularization ==========")
results4 = {}
x_axis4 = list(range(0, 500, 50))
for reg in [0.0, 0.01, 0.1, 0.5]:
    tl, vl, ta, va = train_l2(X_train, Y_train, X_dev, Y_dev, iterations=500, alpha=0.01, lambda_reg=reg, label=f'L2={reg}')
    results4[f'L2={reg}'] = (tl, vl, ta, va)

# Plot Experiment 4
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Experiment 4: Effect of L2 Regularization', fontsize=14)

for key, (tl, vl, ta, va) in results4.items():
    axes[0].plot(x_axis4, tl, label=f'{key} train')
    axes[0].plot(x_axis4, vl, linestyle='--', label=f'{key} val')
    axes[1].plot(x_axis4, ta, label=f'{key} train')
    axes[1].plot(x_axis4, va, linestyle='--', label=f'{key} val')

axes[0].set_title('Loss'); axes[0].set_xlabel('Iteration'); axes[0].legend(fontsize=7)
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Iteration'); axes[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig('experiment4_l2_regularization.png', dpi=150)
plt.show()
print("Saved: experiment4_l2_regularization.png")