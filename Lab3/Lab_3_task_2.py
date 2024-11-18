import matplotlib.pyplot as plt


points = {
    "Class 0": [(0, 0), (1, 1)],
    "Class 1": [(0, 1), (1, 0)]
}

for label, coords in points.items():
    x, y = zip(*coords)
    plt.scatter(x, y, label=label)

x_vals = [0, 1]
y_vals = [0, 1]
plt.plot(x_vals, y_vals, 'r-', label='g(x)')

plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.title("Розділяюча пряма для функції XOR")
plt.xlabel("y1")
plt.ylabel("y2")
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
