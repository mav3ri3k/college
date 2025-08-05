from train import train, test
from model import MLP
from train import init_opt
import flax.nnx as nnx
import prepare_mnist
import util
import jax
from aim import Run
import tomllib
import time

train_data = prepare_mnist.get_data("train")
test_data = prepare_mnist.get_data("test")

with open("./config.toml", "rb") as f:
    data = tomllib.load(f)

num_epoch = data["num_epoch"]
learning_rate = data["learning_rate"]
batch_size = data["batch_size"]
dropout_factor = data["dropout_factor"]

key = jax.random.PRNGKey(0)
model = MLP(dropout_factor, rngs=nnx.Rngs(0))
optimizer = init_opt(model, learning_rate)

start_time = time.time()
for epoch in range(num_epoch):
    train_data = util.shuffle_data(train_data, key)

    for i in range(int(60_000 / batch_size)):
        batch = prepare_mnist.get_batch(train_data, batch_size, i)

        model.train()
        loss = train(model, optimizer, batch)


    model.eval()
    acc = test(model, test_data)
    print(f"Epoch: {epoch:2d}, Accuracy: {acc*100:.2f}")
end_time = time.time()
duration = end_time - start_time
print(f"Trainig took: {duration:2f} seconds")
