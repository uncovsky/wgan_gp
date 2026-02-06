# Image Colorization with Wasserstein GANs

An implementation of WGAN-GP, a Wasserstein GAN with gradient penalty to enforce the Lipschitz smoothness penalty for the critic.
The training scheme and WGAN model are based on the original paper from Arjovsky et al. -- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

Features a U-Net-based generator and a conditional critic, which are defined in wgan_model.py.
The main training loop and losses for the generator and critic, along with monitoring infrastructure (tensorboard logs, checkpointing), are outlined in training.py. 
Specifically, the method fit_gan() trains the model, while generator_loss() and critic_loss() define the two losses.
