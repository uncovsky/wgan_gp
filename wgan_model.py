# STUDENT's UCO: 514479

import torch.nn.functional as F
import torch.nn as nn

from torch import Tensor, cat
import torch
import torchvision


"""
    Convolutional and UpConvolutional blocks for U-net.

    Follow the Ck structure from pix2pix:
        1) Convolution -> InstanceNorm -> Relu
        2) The convolutions are 4x4, stride 2

    `out_channels` specifies the number of filters
    `act_function` specifies the activation function
    `downscale` specifies conv/transposed conv layer


"""

class UNetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 act_function,
                 downscale=True,
                 norm="bn",
                 dropout=False):

        super().__init__()


        # if downscale
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                              stride=2, padding=1, padding_mode='zeros')
        if not downscale:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=4, stride=2, padding=1,
                                           padding_mode='zeros')

        self.act = act_function
        # Stick to 0.5 probability of dropout
        self.dropout = nn.Identity() if not dropout else nn.Dropout2d(p=0.5)
        self.bn = nn.Identity()

        if norm == "bn":
            self.bn = nn.BatchNorm2d(out_channels)
        elif norm == "in":
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        # Conv -> bn -> relu
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        return self.act(x)

"""
    Encoder and Decoder components of the U-net.

    Feature Conv / UpConv blocks that automatically upscale/downscale with
    strides of 2.

    Encoder includes batchnorm on all layers except the input one.
    Decoder also features dropout.


"""
class UNetEncoder(nn.Module):

    def __init__(
                    self,
                    activation_fn,
                    channels = (1, 64, 128, 256, 512, 512, 512, 512, 512),
                    dropouts = (False, False, False, False, False, False, False, False),
                    store_activations = True,
                    batch_norm = True,
                 ) -> None:

        super().__init__()
        layers = []


        # Set normalization settings
        first_layer_norm = None if batch_norm else "in"
        layer_norms = "bn" if batch_norm else "in"

        layers.append(UNetBlock(channels[0], channels[1],
                                activation_fn, downscale=True,
                                dropout=dropouts[0], norm=first_layer_norm))

        for i in range(1, len(channels) - 1):
            layers.append(UNetBlock(channels[i], channels[i+1],
                                    activation_fn, downscale=True,
                                    dropout=dropouts[i], norm=layer_norms))

        # use module list to track layers
        self.layers = nn.ModuleList(layers)
        self.store_activations = store_activations


    # Return features that will be concatenated with the decoder
    # and the final output
    def forward(self, x):
        features = []
        for layer in self.layers[:-1]:
            x = layer(x)

            if self.store_activations:
                features.append(x)

        return self.layers[-1](x), features[::-1]

    def init_weights(self):

        # Change this if activations are changed!
        for layer in self.layers:
                nn.init.kaiming_normal_(layer.conv.weight, nonlinearity='leaky_relu')
                nn.init.constant_(layer.conv.bias, 0.01)


class UNetDecoder(nn.Module):

    def __init__(
                    self,
                    activation_fn,
                    channels = (512, 512, 512, 512, 256, 128, 64, 3),
                    dropouts = (True, True, True, False, False, False, False, False),
                    out_size = (512, 512),
                    batch_norm = True,
                    retain_size = False
                 ) -> None:

        super().__init__()

        layers = []
        self.retain_size = retain_size

        # First layer does not multiply the channels by twice (no concat)
        # Set normalization settings
        layer_norms = "bn" if batch_norm else "in"
        layers.append(UNetBlock(channels[0], channels[1],
                                activation_fn, downscale=False,
                                dropout=dropouts[0], norm=layer_norms))

        for i in range(1, len(channels) - 2):
            layers.append(UNetBlock(2 * channels[i], channels[i+1],
                                    activation_fn, downscale=False,
                                    dropout=dropouts[i], norm=layer_norms))

        # Last layer has Sigmoid to project to [0,1]
        layers.append(UNetBlock(2 * channels[-2], channels[-1],
                                nn.Sigmoid(), downscale=False,
                                dropout=dropouts[-1], norm=layer_norms))

        self.layers = nn.ModuleList(layers)
        self.out_size = out_size

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

    def forward(self, x, enc_features):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)

            # Crop mismatched dimensions
            enc_ftrs = enc_features[i]
            if self.retain_size:
                enc_ftrs = self.crop(enc_features[i], x)

            x = cat([x, enc_ftrs], dim=1)

        result = self.layers[-1](x)

        # Resize to the original image size
        if self.retain_size:
            result = F.Sigmoid(F.interpolate(result, self.out_size))

        return result


    def init_weights(self):
        for layer in self.layers:
                nn.init.kaiming_normal_(layer.conv.weight, nonlinearity='relu')
                nn.init.constant_(layer.conv.bias, 0.01)

class UNet(nn.Module):
    def __init__(self, enc_channels, enc_dropouts,
                       dec_channels, dec_dropouts,
                       out_size, retain_size=False, batch_norm=True):
        super().__init__()

        self.encoder = UNetEncoder(channels=enc_channels,
                                   dropouts=enc_dropouts,
                                   activation_fn=nn.LeakyReLU(negative_slope=0.2),
                                   store_activations=True,
                                   batch_norm=batch_norm)
        self.decoder = UNetDecoder(channels=dec_channels, out_size=out_size,
                                   activation_fn=nn.ReLU(),
                                   retain_size=retain_size,
                                   batch_norm=batch_norm)

        # Make sure to change the weight initializations if activations are changed!
        self.init_weights()


    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()

    def forward(self, x):
        x, enc_features = self.encoder(x)
        res = self.decoder(x, enc_features)
        return res



"""
    WGAN classes

    Wasserstein Conditional Critic:
        input: BW + RGB image, uses pretrained BW encoder along with another
        conv path. These are then combined and passed through a classifier
        head.
"""
class WGAN_Critic(nn.Module):

    def __init__(self, bw_enc_channels, bw_enc_dropouts,
                       rgb_enc_channels, rgb_enc_dropouts,
                       activation_fn,
                       batch_norm = False,
                       head_channels = (1024, 128, 64, 1),
                       head_dropouts = (False, False, False) ):

        super().__init__()

        self.bw_encoder = UNetEncoder(activation_fn, bw_enc_channels,
                                      bw_enc_dropouts,
                                      store_activations=False,
                                      batch_norm=batch_norm)

        self.rgb_encoder = UNetEncoder(activation_fn, rgb_enc_channels,
                                       rgb_enc_dropouts,
                                       store_activations=False,
                                       batch_norm = batch_norm)

        # Classifier Head has RELU activations
        layers = []
        layer_norms = "bn" if batch_norm else "in"
        for i in range(0, len(head_channels) - 1):
            layers.append(UNetBlock(head_channels[i], head_channels[i+1],
                                    nn.ReLU(), downscale=True,
                                    dropout=head_dropouts[i],
                                    norm=layer_norms))

        # Global pooling
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())

        # Linear layers end
        layers.append(nn.Linear(64, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, 1))

        # No activation here, want raw numbers for WGAN critic


        self.critic_head = nn.Sequential(*layers)

        # init weights of both encoders and head
        self.bw_encoder.init_weights()
        self.rgb_encoder.init_weights()
        self.critic_head.apply(self.init_weights_head)

    def forward(self, bw_image, rgb_image):
        l1 = self.bw_encoder(bw_image)[0]
        l2 = self.rgb_encoder(rgb_image)[0]
        # Concat the outputs of the two encoders
        x = cat([l1, l2], dim=1)
        x = self.critic_head(x)

        return x

    def init_weights_head(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.)

        elif isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.)

class WGAN_UNet(nn.Module):
    """
        Conditional Wasserstein GAN for image colorization.

        Generator - UNet
                input_shape ~ [B, 1, H, W]
                output_shape ~ [B, 3, H, W]

                pretrained via L1 loss
        Discriminator -

            Unet encoder for BW image (pretrained, same as generator)
            Another encoder path for RGB
            Followed by more convolutional layers and global pooling

            For an input image, the output is a single scalar, corresponding to
            the dual variable of the Wasserstein distance.
    """

    def __init__(self, gen_config, critic_config, batch_norm_gen, batch_norm_critic):
        super().__init__()

        self.generator_net = UNet(**gen_config, batch_norm=batch_norm_gen)
        self.critic_net = WGAN_Critic(**critic_config, activation_fn = nn.LeakyReLU(negative_slope=0.2), batch_norm=batch_norm_critic)

    # Color image
    def generator(self, bw_image):
        return self.generator_net(bw_image)

    def critic(self, bw_image, rgb_image):
        return self.critic_net(bw_image, rgb_image)

    def forward(self, bw_image):
        rgb_image = self.generator_net(bw_image)
        critic_val = self.critic_net(bw_image, rgb_image)

        return rgb_image, critic_val

    def freeze_critic_encoder(self):
        # Freeze pretrained part of critic
        for param in self.critic_net.bw_encoder.parameters():
            param.requires_grad = False

    def unfreeze_critic_encoder(self):
        for param in self.critic_net.bw_encoder.parameters():
            param.requires_grad = True

    # Load pretrained U-net
    def load_pretrained_generator(self, unet_weights_path):
        weights = torch.load(unet_weights_path)
        self.generator_net.load_state_dict(weights)


    def load_weights(self, load_path):
        weights = torch.load(load_path)
        self.critic_net.bw_encoder.load_state_dict(weights['critic_bw_encoder'])
        self.critic_net.rgb_encoder.load_state_dict(weights['critic_rgb_encoder'])
        self.critic_net.critic_head.load_state_dict(weights['critic_head'])
        self.generator_net.load_state_dict(weights['generator'])

    def save_weights(self, save_path):
        # Save GAN weights, name them accordingly
        torch.save({
                    'critic_bw_encoder': self.critic_net.bw_encoder.state_dict(),
                    'critic_rgb_encoder': self.critic_net.rgb_encoder.state_dict(),
                    'critic_head': self.critic_net.critic_head.state_dict(),
                    'generator': self.generator_net.state_dict(),
                    }, save_path)
