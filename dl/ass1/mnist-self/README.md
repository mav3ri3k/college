# Goals
- [x] Set up your JAX/Flax environment
- [x] Understand the basic components of a neural network
- [x] Train a small MLP on MNIST with Flax (using nnx)

# Learnings
1. Use `git lfs clone` when downloading large files from hugginface
2. Polars > Hugginface Datasets
3. For mnist, even 60k batch size did not meaningfully change my cpu ram usage 🤣
4. Use `rich` for better python tracebacks. I have added it to pythonpath. `uvv` will now give verbose debug traces for better debugging.
