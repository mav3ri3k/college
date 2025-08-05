import jax

def shuffle_data(data, key):
    images = data[0]
    labels = data[1]

    indices = jax.random.permutation(key, images.shape[0])

    return images[indices], labels[indices]
