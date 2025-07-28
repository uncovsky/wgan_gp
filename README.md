# Image Colorization with Wasserstein GANs

An implemenation of WGAN-GP, a Wasserstein GAN with gradient penalty to enforce the Lipschitz smoothness penalty for the critic.

Features an U-net generator, and a conditional critic with two encoders - see network.py.
For the main training loop of WGAN - see training.py.
