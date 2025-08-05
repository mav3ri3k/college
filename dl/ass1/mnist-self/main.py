from train import train, test
from model import MLP
import prepare_mnist
import util
import jax
from aim import Run
import time

train_data = prepare_mnist.get_data("train")
test_data = prepare_mnist.get_data("test")

num_epoch = 10
learning_rate = 1e-3
batch_size = 40

key = jax.random.PRNGKey(0)
model = MLP()
params = model.params(key)

print(f"Hyper-parameters: Epochs: {num_epoch}, Learning Rate: {learning_rate}, Batch Size: {batch_size}")
print()
start_time = time.time()
for epoch in range(num_epoch):
    train_data = util.shuffle_data(train_data, key)

    for i in range(int(60_000 / batch_size)):
        batch = prepare_mnist.get_batch(train_data, batch_size, i)

        loss, params = train(model, batch, learning_rate, params)

        # print(f"Epoch: {epoch:2f}, Batch: {i}, Loss: {loss:2f}")

    acc = test(model, test_data, params)

    print(f"Epoch: {epoch:3d}, Accuracy: {acc*100:2f}")

end_time = time.time()
duration = end_time - start_time
print()
print(f"Trainig took: {duration:2f} seconds")
