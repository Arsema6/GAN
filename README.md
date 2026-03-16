# Generative Art Studio Notebook

This project notebook is a lightweight Conditional GAN (cGAN) studio built with PyTorch and trained on MNIST. It follows the spirit of the 4-day "Generative Art Studio" challenge by combining fast training, visual progress tracking, and class-controlled image generation in a notebook-friendly workflow.

## Project Summary

The notebook currently trains a convolutional conditional GAN on handwritten digits. The generator takes both random noise and a class label, while the discriminator evaluates whether an image-label pair is real or fake. This allows the model to learn both image structure and class-specific control.

Current notebook behavior:

- Uses MNIST as the dataset.
- Resizes images to `32x32` for fast GPU-friendly training.
- Trains a convolutional generator and discriminator.
- Conditions both models on digit labels.
- Stores generator snapshots after each epoch.
- Produces a control example for digit `8` generation.

## Challenge Alignment

### Day 1: Choose Your Muse

This notebook uses MNIST, which is a strong choice for a fast experiment because:

- it downloads automatically,
- trains quickly on free GPU resources,
- makes GAN progress easy to see,
- and works well for conditional generation.

Recommended environment:

- Google Colab or Kaggle Notebook
- GPU enabled in runtime settings
- PyTorch with torchvision and matplotlib installed

## Notebook Pipeline

The notebook is organized around the following stages:

1. Import PyTorch, torchvision, and plotting utilities.
2. Detect whether CUDA is available.
3. Load and normalize the MNIST training set.
4. Define the conditional convolutional generator.
5. Define the conditional convolutional discriminator.
6. Train the GAN for several epochs.
7. Save visual snapshots of generated samples over time.
8. Generate class-specific outputs, including multiple samples of digit `8`.

## Architecture Notes

### Generator

The generator:

- starts from a random latent vector,
- embeds the class label,
- concatenates noise and label information,
- then upsamples into an image using transposed convolutions.

### Discriminator

The discriminator:

- receives an image and its label,
- expands the label embedding across the image size,
- concatenates the image and label channels,
- then predicts whether the pair is real or generated.

This is a standard conditional DCGAN-style setup and is a good fit for the challenge requirement of "mastering control."

## Outputs Already Covered

### 1. Evolution Map

The notebook already stores generator outputs during training and displays them as a grid across epochs. This gives a simple visual record of how random noise gradually becomes digit-like structure.

### 2. Control Grid

The notebook already demonstrates label conditioning by generating multiple samples of a chosen class. In the current version, it creates several versions of digit `8` from different random seeds.

## Good Next Extensions

To fully match the challenge brief, the following additions are recommended:

### 1. Morphing Reel

Interpolate between two latent vectors and generate intermediate images. This will show whether the learned latent space changes smoothly between samples.

### 2. Finer Snapshot Tracking

Instead of saving only one snapshot per epoch, store outputs every fixed number of training steps such as every `1000` iterations.

### 3. Performance Audit

Add:

- `torch.cuda.max_memory_allocated()` to compare memory use,
- an activation comparison such as `LeakyReLU` vs `GELU`,
- and a short visual check for checkerboard artifacts from transposed convolutions.

### 4. Failure Insight

Include a short written note describing one bad generation example and what it suggests about the model. For MNIST, this could be:

- incomplete loops in `8`,
- merged strokes between digits,
- or ambiguous samples that resemble two classes at once.

## Suggested Deliverables

For a complete submission, include:

- an evolution grid across training,
- a control grid showing diverse versions of one requested class,
- a latent interpolation grid,
- and a short reflection on one generation failure.

## Practical Tips

- Save model checkpoints every few epochs using `.pth` files.
- If you are using Colab, save checkpoints to Google Drive.
- Keep image sizes small for fast experiments.
- Use fixed noise vectors when comparing training progress across epochs.

## Conclusion

This notebook is already a solid starting point for the Generative Art Studio challenge. It covers the core cGAN workflow, shows training evolution, and supports prompt-like control through class labels. With latent interpolation, memory tracking, and a short failure analysis added, it can become a complete mini generative art portfolio.
