import numpy as np
import tensorflow as tf

# Define parameters
n_samples, batch_size, num_steps = 1000, 100, 20000
X_data = np.random.uniform(1, 10, (n_samples, 1)).astype(np.float32)  # Ensure float32
y_data = (2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))).astype(np.float32)  # Ensure float32

# Use tf.Variable for parameters
k = tf.Variable(tf.random.normal((1, 1), dtype=tf.float32), name='slope')
b = tf.Variable(tf.zeros((1,), dtype=tf.float32), name='bias')

# Define the loss function and optimizer
def compute_loss(X, y):
    y_pred = tf.matmul(X, k) + b
    return tf.reduce_mean((y - y_pred) ** 2)

optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Training loop
display_step = 100
for i in range(num_steps):
    indices = np.random.choice(n_samples, batch_size)
    X_batch = X_data[indices].astype(np.float32)  # Convert to float32
    y_batch = y_data[indices].astype(np.float32)  # Convert to float32

    # Use GradientTape for gradient computation
    with tf.GradientTape() as tape:
        loss = compute_loss(X_batch, y_batch)

    gradients = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(gradients, [k, b]))

    if (i + 1) % display_step == 0:
        print(f"Step {i + 1}: Loss = {loss.numpy():.8f}, k = {k.numpy()[0, 0]:.4f}, b = {b.numpy()[0]:.4f}")
