# MNIST Conditional GAN Notebook

The notebook walks through a complete cGAN workflow:

1. Imports the required PyTorch, torchvision, and matplotlib libraries.
2. Detects whether a GPU is available and selects `cuda` or `cpu`.
3. Loads the MNIST training set with resizing, tensor conversion, and normalization.
4. Defines a conditional generator based on transposed convolution layers.
5. Defines a conditional discriminator based on convolution layers.
6. Trains both models adversarially for several epochs.
7. Stores generated image snapshots at the end of each epoch.
8. Displays a training-evolution grid.
9. Generates a label-controlled sample grid for a chosen digit.

## Dataset And Preprocessing

The notebook uses the MNIST handwritten digits dataset.

Preprocessing steps:

- images are resized to `32x32`,
- converted to tensors,
- and normalized to the range expected by a `Tanh` output layer.

The data loader uses shuffled mini-batches, which is appropriate for GAN training.

## Model Architecture

### Generator

The generator takes two inputs:

- a random latent vector of size `100`,
- and a class label.

The label is embedded, concatenated with the latent vector, reshaped into a feature map, and passed through several `ConvTranspose2d` blocks with batch normalization and `ReLU` activations. The final layer uses `Tanh` to produce a single-channel image.

This allows the generator to create digits conditioned on a requested class.

### Discriminator

The discriminator also uses label conditioning.

It embeds the class label, expands it spatially to match the image dimensions, concatenates it with the input image, and passes the result through convolutional layers with `LeakyReLU`, batch normalization, and a final `Sigmoid` output.

Its job is to decide whether an image-label pair is real or generated.

## Training Procedure

The notebook trains the model for `8` epochs.

For each batch:

- the discriminator is trained on real images with true labels,
- the discriminator is trained on fake images generated from random noise and random labels,
- then the generator is updated to fool the discriminator.

The loss function used is binary cross-entropy loss (`BCELoss`), and both models are optimized with Adam using GAN-friendly beta values.

At the end of every epoch, the notebook generates samples from a fixed noise tensor and fixed labels, then stores them in a `snapshots` list. This makes it possible to compare model progress consistently across training.

## Visual Outputs

The notebook produces two main visual results.

### 1. Training Evolution Grid

After training, the notebook plots one generated image grid per epoch. This shows how the generator evolves from noisy outputs toward more recognizable digit shapes.

### 2. Label-Controlled Generation

The notebook then generates a new batch of samples using a chosen label, specifically digit `8`, and displays them in a grid. This demonstrates that the model has learned conditional generation rather than only unconditional sampling.

## Notebook Structure

The notebook is organized as a straightforward sequence of code cells:

- library imports,
- device setup,
- data transforms and dataset loading,
- generator definition,
- discriminator definition,
- hyperparameter and optimizer setup,
- fixed latent vectors for snapshot tracking,
- training loop,
- evolution-grid visualization,
- class-controlled generation,
- final image display.

## Requirements

To run the notebook, you need:

- Python
- PyTorch
- torchvision
- matplotlib

A GPU is recommended but not strictly required.

## Current Strengths

This notebook already has several good design choices:

- conditional generation is implemented correctly in both models,
- convolutional layers are used instead of a fully connected GAN,
- fixed noise is used for consistent snapshot comparison,
- and the workflow is compact enough for Colab or Kaggle.

## Possible Improvements

If you want to extend this notebook further, the most natural next additions are:

- save snapshots every fixed number of iterations rather than only per epoch,
- interpolate between two latent vectors to explore the latent space,
- save model weights as `.pth` checkpoints,
- measure GPU memory with `torch.cuda.max_memory_allocated()`,
- and compare different activation functions such as `LeakyReLU` and `GELU`.

## Summary

This notebook is a compact MNIST cGAN implementation that covers dataset loading, conditional GAN architecture, adversarial training, training-progress visualization, and class-controlled image generation. It is a solid foundation for experimenting with generative art ideas in a lightweight notebook setting.
