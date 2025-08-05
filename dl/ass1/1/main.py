import random
import matplotlib.pyplot as plt

random.seed(42)

class Perceptron:
    def __init__(self, lr=1.0):
        self.w0 = random.uniform(-1, 1)
        self.w1 = random.uniform(-1, 1)
        self.w2 = random.uniform(-1, 1)
        self.lr = lr

    def predict(self, x):
        z = self.w0 + self.w1 * x[0] + self.w2 * x[1]
        return 1 if z >= 0 else -1

    def update(self, x, target):
        pred = self.predict(x)
        if pred != target:
            self.w0 += self.lr * target * 1.0
            self.w1 += self.lr * target * x[0]
            self.w2 += self.lr * target * x[1]
            return True
        return False

x = [[2, 3], [1, 4], [2, 4], [4, 2], [5, 1], [4, 3]]
y = [+1, +1, +1, -1, -1, -1]

model = Perceptron(lr=1.0)

max_epochs = 100
epoch_errors = []

plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('Mis-classifications')
ax.set_title('Perceptron convergence on EEG data')
ax.grid(True)
line, = ax.plot([], [], marker='o')
plt.show()

for epoch in range(max_epochs):
    miss = 0
    for xi, yi in zip(x, y):
        if model.update(xi, yi):
            miss += 1
    epoch_errors.append(miss)

    line.set_data(range(len(epoch_errors)), epoch_errors)
    ax.relim(); ax.autoscale_view()
    plt.pause(0.2)

    print(f"[epoch {epoch:02d}] miss={miss}")
    if miss == 0:
        print("Converged!")
        break

plt.ioff()
plt.show()

print(f"Final bias: {model.w0}, Weights: {model.w1}, {model.w2}")
