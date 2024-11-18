import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


np.random.seed(1)
weights_input_hidden = np.random.uniform(size=(2, 2))
weights_hidden_output = np.random.uniform(size=(2, 1))
bias_hidden = np.random.uniform(size=(1, 2))
bias_output = np.random.uniform(size=(1, 1))


learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate


plt.figure(figsize=(8, 6))
plt.title("Двошаровий перцептрон для XOR")

plt.scatter(-1, 2, s=500, c='cyan', label='Вхід x1')
plt.scatter(-1, 0, s=500, c='cyan', label='Вхід x2')

plt.scatter(1, 1, s=500, c='orange', label='Прихований нейрон h1')
plt.scatter(1, -1, s=500, c='orange', label='Прихований нейрон h2')

plt.scatter(3, 0, s=500, c='pink', label='Вихід')

plt.plot([-1, 1], [2, 1], 'k--')
plt.plot([-1, 1], [2, -1], 'k--')
plt.plot([-1, 1], [0, 1], 'k--')
plt.plot([-1, 1], [0, -1], 'k--')
plt.plot([1, 3], [1, 0], 'k--')
plt.plot([1, 3], [-1, 0], 'k--')

plt.legend(loc="best")
plt.axis('off')
plt.show()
