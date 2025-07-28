# STUDENT's UCO: 514479

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <dataset_path>

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from dataset import PlacesDataset, load_validate_images, train_test_split
from wgan_model import UNet, UNetEncoder, UNetDecoder, WGAN_UNet

from torch import Tensor, nn
from torch.optim import Optimizer

from torchsummary import summary

from torch.utils.data import DataLoader
from torch.linalg import vector_norm

class ValidationTracker:

    def __init__(self, model_dir = "models/", top_n=3,
                       img_every_n=10, num_images=30,
                       log_dir = "logs/", early_stop=False, early_stop_n=5):

        # Create path to save best performing models.
        output_path = Path(model_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model_dir = model_dir
        self.writer = SummaryWriter(log_dir=log_dir)

        self.top_n = top_n
        self.top_n_losses = []

        self.not_improved_steps = 0
        self.early_stop = early_stop
        self.early_stop_n = early_stop_n

        self.num_images = num_images
        self.img_every_n = img_every_n

        self.critic_losses = []
        self.generator_losses = []

        self.critic_losses_val = []
        self.generator_losses_val = []


    def get_losses(self):
        return self.train_losses, self.val_losses


    def log_scalar(self, value, step, label, log=True, generator=False):
        """
            Logs training losses
        """
        self.writer.add_scalar(label, value, global_step=step)

        if log:
            if generator:
                self.generator_losses.append(value)
            else:
                self.critic_losses.append(value)


    # Evaluates model on validation dataset, returns its loss
    # If save_images flag is set, saves first num_image predictions
    def validate_model(self, model, validation_data, device, save_images=False):

        gen_validation_loss = 0.0
        critic_validation_loss = 0.0

        # Critic loss on fake samples vs real
        fake_vals = 0.0
        real_vals = 0.0

        images = []
        labels = []

        model.eval()

        for data in validation_data:

            bw_image = data[0].to(device)
            rgb_image = data[1].to(device)
            label = data[2][0]

            with torch.no_grad():
                fake_image = model.generator(bw_image)

                # calculate losses (no gradient penalty)
                values_fake, values_real, gp = critic_loss(bw_image, rgb_image,
                                          model, 0)

                loss_critic = values_fake - values_real
                loss_gen = generator_loss(bw_image, rgb_image, model)

            if save_images and len(images) < self.num_images:

                images.append(fake_image.squeeze(0).cpu())
                images.append(rgb_image.squeeze(0).cpu())

                labels.append(f"{label}_prediction")
                labels.append(f"{label}_truth")

            critic_validation_loss += loss_critic.detach().item()
            gen_validation_loss += loss_gen.detach().item()
            fake_vals += values_fake.detach().item()
            real_vals -= values_real.detach().item()


        gen_validation_loss /= len(validation_data)
        critic_validation_loss /= len(validation_data)
        fake_vals /= len(validation_data)
        real_vals /= len(validation_data)

        return  gen_validation_loss, critic_validation_loss, fake_vals, \
                real_vals, images, labels


    # Save models with best critic loss, values of critic loss (without grad
    # norm penalty) correspond to estimates of WD

    def add_loss(self, model, loss):
        if len(self.top_n_losses) < self.top_n:
            self.top_n_losses.append(loss)
            self.top_n_losses.sort()
            return True

        # Save top n models
        for i, l in enumerate(self.top_n_losses):
            if loss < l:
                self.top_n_losses[i] = loss
                model.save_weights(self.model_dir + f"model{i}.pt")
                return True

        # Return false if not improved over top_n
        return False


    def __call__(self, epoch, model, validation_data, device):

        """
            Callback function that aggregates all the above steps, called after every
            epoch.
        """

        save_images = epoch % self.img_every_n == 0

        # Validate model
        gen_loss, critic_loss, fake_vals, real_vals, images, labels = self.validate_model(model,
                                                                      validation_data,
                                                                      device,
                                                                      save_images)
        # Save model if it is in `top_n` best models
        if self.add_loss(model, critic_loss):
            print(f"New top {self.top_n} model on validation set!")

        else:
            self.not_improved_steps += 1

        # If the models have not improved for `early_stop_n` epochs, stop
        if self.early_stop and self.not_improved_steps >= self.early_stop_n:
            print("Early stopping triggered.")
            return True

        # Log the loss
        self.writer.add_scalar("[Validation] Gen Loss", gen_loss, global_step=epoch)
        self.writer.add_scalar("[Validation] Wasserstein Estimate", -1 * critic_loss, global_step=epoch)
        self.writer.add_scalar("[Validation] Critic Loss (real)", real_vals, global_step=epoch)
        self.writer.add_scalar("[Validation] Critic Loss (fake)", fake_vals, global_step=epoch)

        self.generator_losses_val.append(gen_loss)
        self.critic_losses_val.append(critic_loss)

        # Log image grid
        if save_images:
            self.writer.add_figure(f"Prediction for {self.num_images} of current model",
                                   image_grid(images, labels), global_step=epoch)
        return False

"""
    Model configs, etc.
"""

COL_CHANNELS = 3
IMG_SIZE = (512, 1024)


U_NET_SMALL = {
        "enc_channels" : (1, 32, 64, 128, 256, 512),
        "dec_channels" : (512, 256, 128, 64, 32, COL_CHANNELS),
        "enc_dropouts" : (False, False, False, False, False, False),
        "dec_dropouts" : (True, True, True, False, False, False),
        "out_size" : IMG_SIZE,
}

WGAN_CRITIC_PRETRAINED = {
        "bw_enc_channels" : (1, 32, 64, 128, 256),
        "bw_enc_dropouts" : U_NET_SMALL["enc_dropouts"],
        "rgb_enc_channels" : (3, 32, 64, 128, 256),
        "rgb_enc_dropouts" : U_NET_SMALL["enc_dropouts"],
        "head_channels" : (512, 256, 128, 64),
        "head_dropouts" : (False, False, False, False)
}

# Load pretrained generator, smaller critic
WGAN_PRETRAINED = {
    "gen_config" : U_NET_SMALL,
    "critic_config" : WGAN_CRITIC_PRETRAINED,
    "batch_norm_critic" : False,
    "batch_norm_gen" : True,
}

"""
    Helper functions for visualization, etc.
"""

def parameter_count(model):
    total = 0
    for param in model.parameters():
        param_size = 1
        for dim in param.size():
            param_size *= dim
        total += param_size
    return total


def image_grid(images, titles, grayscale=False):

    num_images = len(images)
    num_cols = 2
    num_rows = (num_images + 1) // 2

    cmap = 'gray' if grayscale else None

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 4 * num_rows))
    axes = axes.flatten() if num_images > 1 else [axes]

    # Undo normalization

    for idx, image in enumerate(images):
        ax = axes[idx]
        img = image.permute(1, 2, 0)
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
        if titles and idx < len(titles):
            ax.set_title(titles[idx])

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig




"""
    TRAINING
        Loss functions
"""

def generator_loss(bw_images, rgb_images, gan, lmbda=10):

    # Generate fake images
    fake_images = gan.generator(bw_images)

    # Get their values
    values = gan.critic(bw_images, fake_images)

    # Maximize the values of generated images
    loss = -values.mean()

    return loss



def critic_loss(bw_images, rgb_images, gan, lmbda=10):
    """
        bw_images / rgb_images: batch of images from real data
        fake_images - generated via gan.generator
        gan : WGAN model
        lmbda : lagrangian for gradient norm penalty
    """

    # Estimate of Wasserstein Distance
    with torch.no_grad():
        fake_images = gan.generator(bw_images)

    values_real = gan.critic(bw_images, rgb_images)
    values_fake = gan.critic(bw_images, fake_images)


    # If no gradient penalty, skip calculating the gradients
    if lmbda == 0:
        return values_fake.mean(), values_real.mean(), torch.tensor([0.]).to(values_fake.device)

    # penalize gradient norm
    batch_size = bw_images.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1).to(rgb_images.device)

    # interpolate between real and fake images
    interp = (alpha * rgb_images + (1 - alpha) * fake_images).requires_grad_(True)
    interp_values = gan.critic(bw_images, interp)

    # Calculate gradients at interpolated points
    grads = torch.autograd.grad(
        outputs=interp_values,
        inputs=interp,
        grad_outputs=torch.ones_like(interp_values),
        create_graph=True,
        retain_graph=True)[0]

    # Reshape into a 2d matrix and take norms
    norm = vector_norm(grads.view(batch_size, -1), dim=1)
    grad_penalty = ((norm - 1) ** 2).mean() * lmbda

    return values_fake.mean(), values_real.mean(), grad_penalty


"""
    TRAINING
        Training loop
"""
def fit_gan(
    gan: nn.Module,
    batch_size: int,
    epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer_critic: Optimizer,
    optimizer_generator : Optimizer,
    device: torch.device,
    critic_iterations = 5,
    gen_frozen_epochs = 0,
    lmbda=10,
    weight_clip=0.05
) -> tuple[list[float], list[float]]:

    running_loss = 0.0

    gen_losses = []
    critic_losses = []

    for epoch in range(1, epochs + 1):

        # Three components of critic loss 
        critic_real_loss = 0.0
        critic_fake_loss = 0.0
        grad_penalty = 0.0

        generator_training_loss = 0.0

        gan.train()

        for i, data in enumerate(train_dataloader):

            bw_images = data[0].to(device)
            rgb_images = data[1].to(device)

            fake_vals, real_vals, gp = critic_loss(bw_images, rgb_images, gan, lmbda)


            optimizer_critic.zero_grad()
            # Critic estimates wasserstein dist
            critic_loss_value = fake_vals - real_vals + gp
            critic_loss_value.backward()
            optimizer_critic.step()

            grad_penalty += gp.detach().item()

            critic_real_loss -= real_vals.detach().item()
            critic_fake_loss += fake_vals.detach().item()


            # Every `critic_iterations` generator moves as well.
            if epoch > gen_frozen_epochs and i % critic_iterations == 0:

                optimizer_generator.zero_grad()
                gen_loss_value = generator_loss(bw_images, rgb_images, gan)
                gen_loss_value.backward()
                optimizer_generator.step()

                generator_training_loss += gen_loss_value.detach().item()


        generator_training_loss /= len(train_dataloader)
        grad_penalty /= len(train_dataloader)
        critic_real_loss /= len(train_dataloader)
        critic_fake_loss /= len(train_dataloader)

        print("[GEN] Train loss: ", generator_training_loss)
        print("[CRIT] Train loss: ", critic_real_loss + critic_fake_loss)

        gen_losses.append(generator_training_loss)
        critic_losses.append(critic_real_loss + critic_fake_loss)



    print("Training finished!")

    return gen_losses, critic_losses


# declaration for this function should not be changed
def training(dataset_path: Path) -> None:

    """Performs training on the given dataset.

    Args:
        dataset_path: Path to the dataset.

    Saves:
        - model.pt (trained model)
        - learning_curves.png (learning curves generated during training)
        - model_architecture.png (a scheme of model's architecture)
    """
    # Check for available GPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing with {}!".format(device))

    batch_size = 1

    print("Loading \& validating images...")
    df = load_validate_images(dataset_path)
    print(" Validation successful.")


    """
        Dataset Creation
        Validation split
    """
    train_data, validation_data = train_test_split(df, test_size=0.2)
    validation_data, test_data = train_test_split(validation_data, test_size=0.2)

    train_dataset = PlacesDataset(train_data)
    validation_dataset = PlacesDataset(validation_data)
    test_dataset = PlacesDataset(validation_data)

    train_dataloader = DataLoader( train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True )

    # Setting shuffle=True here, since I'll be using the validation set to
    # visualize images. I'll use ~10 images, want random ones.
    val_dataloader = DataLoader( validation_dataset,
                                 batch_size=1,
                                 shuffle=True )

    test_dataloader = DataLoader( test_dataset,
                                 batch_size=1,
                                 shuffle=True )

    """
        Initialize the model
    """
    gan = WGAN_UNet(**WGAN_PRETRAINED).to(device)

    print("Total GAN Parameters:", parameter_count(gan))

    """
        Save model architecture
    """
    input_sample = torch.zeros((1, 1, 512, 1024))

    # define optimizer and learning rate
    # reduce momentum for critic as per authors observations
    crit_optimizer = torch.optim.Adam(gan.critic_net.parameters(), lr=5e-4, betas=(0.5,
                                                                    0.999))
    gen_optimizer = torch.optim.Adam(gan.generator_net.parameters(), lr=1e-4,
                                     betas=(0.5, 0.999))

    num_epochs = 1

    # train the network
    gen_losses, critic_losses = fit_gan(
        gan, batch_size, num_epochs, train_dataloader, val_dataloader,
        crit_optimizer, gen_optimizer, device,
    )

    plot_learning_curves(gen_losses, critic_losses, "Training Loss")
    gan.save_weights("model.pt")



# #### code below should not be changed ############################################################################


def main() -> None:
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()
    training(args.dataset_path)


if __name__ == "__main__":
    main()


