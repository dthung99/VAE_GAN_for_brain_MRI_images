"""This file contains the architectures and functions that I used in the project (.ipynb notebook)"""
"""Its purpose is mainly for testing and validating each one seperately"""
# Import library and set up environment
# General library
import os as os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from collections import OrderedDict
import time
# Specific library needed for the project
import nibabel as nib

# Define Swish as an activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# PositionalEncoding for multi-head attention
class PositionalEncoding(nn.Module):
    """I copy it here https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch"""
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 50000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Define a resdidual block
class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, mode="level", activation_func=nn.LeakyReLU):
    '''
      in_channels - integer - the number of feature channels the first
                              convolution will receive
      out_channels - integer - the number of feature channels the last
                               convolution will output

      mode - string - defines what the block will do
           - "upsample" means the block wil double the spatial size
           - "downsample" means the block will halve the spatial size
           - "level" means the block will not change the spatial dimension
    '''
    super().__init__()
    assert kernel_size%2 == 1, "please use odd kernel_size"
    assert type(activation_func) == type, "activation_func should be a class"
    assert type(activation_func) == type, f"activation_func {activation_func} should be a class"
    padding = (kernel_size-1)//2 # Padding for each kernel
    if mode == "upsample":
      self.main_forward = nn.Sequential(
          nn.ConvTranspose2d(
              in_channels=in_channels,
              out_channels=out_channels,
              kernel_size=4,
              stride=2,
              padding=1,
          ),
          nn.BatchNorm2d(out_channels),
          activation_func(),
          nn.Dropout(),
          nn.Conv2d(
              in_channels=out_channels,
              out_channels=out_channels,
              kernel_size=kernel_size,
              stride=1,
              padding=padding,
          ),
      )
      self.residual_forward = nn.ConvTranspose2d(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=4,
          stride=2,
          padding=1,
      )
    else:
      self.main_forward = nn.Sequential(
          nn.Conv2d(
              in_channels=in_channels,
              out_channels=out_channels,
              kernel_size=kernel_size,
              stride=2 if mode == "downsample" else 1,
              padding=padding,
          ),
          nn.BatchNorm2d(out_channels),
          activation_func(),
          nn.Dropout(),
          nn.Conv2d(
              in_channels=out_channels,
              out_channels=out_channels,
              kernel_size=kernel_size,
              stride=1,
              padding=padding,
          ),
      )
      self.residual_forward = nn.Conv2d(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=kernel_size,
          stride=2 if mode == "downsample" else 1,
          padding=padding,
      )

  def forward(self, x):
    out = self.main_forward(x)
    x = out + self.residual_forward(x)
    return x

# Define a multilayer perceptron
class MLP(nn.Module):
    def __init__(self, number_of_features, activation_func=nn.LeakyReLU):
        super(MLP, self).__init__()
        assert type(number_of_features) == list, "MLP input is list"
        assert len(number_of_features) > 1, "MLP input length is not good"
        assert type(activation_func) == type, f"activation_func {activation_func} should be a class"

        for num in number_of_features:
            assert type(num) == int, "MLP input should be int"
        # Dict that store the layers
        linear_dict = OrderedDict()
        i = -1
        for i in range(len(number_of_features) - 2):
            linear_dict[f"Linear_layer_{i+1}"] = nn.Linear(
                in_features = number_of_features[i],
                out_features = number_of_features[i+1],
            )
            linear_dict[f"Batch_norm_{i+1}"] = nn.BatchNorm1d(number_of_features[i+1])
            linear_dict[f"Relu_{i}"] = activation_func()
            linear_dict[f"Dropout_{i}"] = nn.Dropout()

        i+=1
        # Last layer
        linear_dict[f"Linear_layer_{i+1}"] = nn.Linear(
            in_features = number_of_features[i],
            out_features = number_of_features[i+1],
        )
        # Combine into one
        self.all_linear = nn.Sequential(linear_dict)
    def forward(self, x):
        return self.all_linear(x)
    
# DEFINE YOUR ENCODER ARCHITECTURE MODEL TO PREDICT BRAIN AGE
class Encoder(nn.Module):
    def __init__(self, in_channels=1, image_size=[196, 230], depth=4, length=1, complexity=32, latent_channels=32*4,
                 activation_func=nn.LeakyReLU, attention_dropout=0.1):
        """
        in_channels - integer: number of channels of input image
        image_size - tuple: size of image
        depth - integer: how many time the network downsample the image
        length - integr: how many CNN will the image underwent before downsampled,
        latent_size - integer: length of latent vector,
        complexity - integer: no of channels of first convolution
        """
        super().__init__()
        # Assert that the image could be downsampled with given depth
        image_size = torch.tensor(image_size)
        assert torch.log2(image_size).min() > depth, "The network might not be able to downsample the image to such depth"
        assert type(activation_func) == type, f"activation_func {activation_func} should be a class"
        # Declare the dict for encoder
        encoder = OrderedDict()
        # First layer
        feature_size = in_channels*complexity
        # For each depth, the network go into a length number of resnet before downsampled
        for d in range(0, depth, 1):
            # Reset in_channels and doubling the feature_size
            # Extract more features in this depth
            for l in range(0, length - 1, 1):
                encoder["encoder-depth_"+str(d)+"-length_"+str(l)] = nn.Sequential(
                    ResBlock(
                        in_channels = in_channels,
                        out_channels = in_channels,
                        kernel_size = 3,
                        mode = "level",
                        activation_func = activation_func,
                    ),
                    nn.BatchNorm2d(in_channels),
                    activation_func(),
                    nn.Dropout(),
                )
            # Downsampling
            encoder["encoder-depth_"+str(d)+"-downsample"] = nn.Sequential(
                ResBlock(
                    in_channels = in_channels,
                    out_channels = feature_size,
                    mode = "downsample",
                    activation_func = activation_func,
                ),
                nn.BatchNorm2d(feature_size),
                activation_func(),
                nn.Dropout(),
            )
            in_channels = feature_size
            feature_size *= 2

        self.encoder = nn.Sequential(encoder)
        # Last layer to bottleneck the network
#         feature_size = in_channels
#         in_channels = feature_size
        feature_size = latent_channels
        self.last_bottleneck_layer = nn.Sequential(
            ResBlock(
                in_channels = in_channels,
                out_channels = feature_size,
                kernel_size = 3,
                mode = "level",
                activation_func = activation_func,
            ),
        )

        self.output_size = torch.cat([torch.tensor([feature_size]), torch.ceil(image_size/2**depth)], dim = 0)
        self.output_size = self.output_size.to(torch.int32)

    def forward(self, x):
        x = self.encoder(x)
        x = self.last_bottleneck_layer(x)
        return x
    
# DEFINE YOUR ENCODER ARCHITECTURE MODEL TO PREDICT BRAIN AGE
class EncoderClassifier(nn.Module):
    def __init__(self, in_channels=1, image_size=[196, 230],
                 depth=4, length=1, complexity = 32, latent_channels=32*4,
                 classifer_dim=[64, 1], activation_func=nn.LeakyReLU, attention_dropout=0.1):
        """
        in_channels - integer: number of channels of input image
        image_size - tuple: size of image
        depth - integer: how many time the network downsample the image
        length - integr: how many CNN will the image underwent before downsampled,
        classifer_dim - list: dimension of hidden and last layer
        """
        assert type(activation_func) == type, f"activation_func {activation_func} should be a class"
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, image_size=image_size, depth=depth, length=length, complexity=complexity,
                               latent_channels=latent_channels, activation_func=activation_func, attention_dropout=attention_dropout)
        self.bn_and_activation = nn.Sequential(
            nn.BatchNorm2d(int(self.encoder.output_size[0])),
            activation_func(),
            nn.Dropout(),
        )
        if len(classifer_dim)>1:
            self.classifier = nn.Sequential(
                MLP([int(self.encoder.output_size.prod()), *classifer_dim[:-1]], activation_func=activation_func),
                nn.BatchNorm1d(classifer_dim[-2]),
                activation_func(),
                nn.Dropout(),
                nn.Linear(classifer_dim[-2],classifer_dim[-1]),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(int(self.encoder.output_size.prod()),classifer_dim[-1]),
            )
            
        # Initiate the weights
        self.initialize_weights()
        
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                                   nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                                   nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            
    def forward(self, x):
        x = torch.flatten(self.bn_and_activation(self.encoder(x)), start_dim = 1)
        return self.classifier(x)

# ReLU but with custom cut-off
class CustomReLU(nn.Module):
    """Last layer of decoder
    Make all pixels less than the zero_pixel_after_transform become zero_pixel_after_transform"""
    def __init__(self, cut_off = 0):
        super().__init__()
        self.cut_off = torch.tensor(cut_off)
    def forward(self, x):
        x = x-self.cut_off
        return x*(x>0)+self.cut_off

# MSE loss with mask
class MaskedMSELoss(nn.Module):
    def __init__(self, cut_off = 0, non_zero_weight = 1.0, zero_weight = 0.0):
        super().__init__()
        self.cut_off = torch.tensor(cut_off)
        self.non_zero_weight = non_zero_weight
        self.zero_weight = zero_weight

    def forward(self, output, target):
        """
        Calculates the mse loss function, where 0 pixels is not consider in the mean.
        Args:
            output (torch.Tensor): The model's output.
            target (torch.Tensor): The ground truth target.
        Returns:
            torch.Tensor: The calculated loss.
        """
        loss = (output-target)**2
        loss_non_zero = loss[target>self.cut_off]
        loss_zero = loss[target<=self.cut_off]
        if len(loss_zero) == 0:
            loss_zero = torch.tensor([0.0])
        if len(loss_non_zero) == 0:
            loss_non_zero = torch.tensor([0.0])
        return self.non_zero_weight*loss_non_zero.mean() + self.zero_weight*loss_zero.mean()
   
# MSE loss with mask
class MaskedMSELoss_Simple(nn.Module):
    def __init__(self, cut_off = 0):
        super().__init__()
        self.cut_off = torch.tensor(cut_off)

    def forward(self, output, target):
        """
        Simplier version to speed up initial training
        Calculates the mse loss function, where 0 pixels is not consider in the mean.
        Args:
            output (torch.Tensor): The model's output.
            target (torch.Tensor): The ground truth target.
        Returns:
            torch.Tensor: The calculated loss.
        """
        loss = (output-target)**2
        loss_non_zero = loss[target>self.cut_off]
        return loss_non_zero.mean()
    
# DEFINE YOUR DECODER ARCHITECTURE MODEL TO RECONSTRUCT THE IMAGE
class Decoder(nn.Module):
    def __init__(self, in_channels=512, input_image_size=[25, 29], output_image_size=[196, 230], depth=4, length=1, complexity=32,
                 out_channels=1, activation_func=nn.LeakyReLU, attention_dropout=0.1, zero_pixel_after_transform = 0):
        """
        in_channels - integer: number of channels of input image
        image_size - tuple: size of image
        depth - integer: how many time the network downsample the image
        length - integr: how many CNN will the image underwent before downsampled,
        latent_size - integer: length of latent vector,
        complexity - integer: no of channels of first convolution
        """
        super().__init__()
        # Assert that the image could be downsampled with given depth
        output_image_size = torch.tensor(output_image_size)
        assert torch.log2(output_image_size).min() > depth, "The network might not be able to downsample the image to such depth"
        assert type(activation_func) == type, f"activation_func {activation_func} should be a class"
        # Store the input size
        self.input_image_size = [int(i) for i in input_image_size]
        # Declare the dict for encoder
        decoder = OrderedDict()
                
        # First layer
        feature_size = complexity*2**depth
        self.first_layer_to_increase_no_channels = nn.Sequential(
            ResBlock(
                in_channels = in_channels,
                out_channels = feature_size,
                kernel_size = 3,
                mode = "level",
                activation_func = activation_func,
            ),
            nn.BatchNorm2d(feature_size),
            activation_func(),
            nn.Dropout(),
        )
        
        # Attention layer to introduce some spatial variance for a network with all conv layer
        # The output of the conv layer is (bacth, feature_size, *image_size)
        # The input of the attention layer is (bacth, seq_size = image_size, embed_dim = feature_size)
        self.first_positional_encoding = PositionalEncoding(embed_dim=feature_size, dropout=attention_dropout)
        self.first_multiheadattention = nn.MultiheadAttention(embed_dim=feature_size,
                                                              num_heads=feature_size//complexity,
                                                              batch_first=False,
                                                              dropout=attention_dropout)
        self.activation_after_multiheadattention_and_reshape = nn.Sequential(
            nn.BatchNorm2d(feature_size),
            activation_func(),
            nn.Dropout(),
        )

        
        # For each depth, the network go into a length number of resnet before downsampled
        for d in range(0, depth, 1):
            # Reset in_channels and doubling the feature_size
            in_channels = feature_size
            feature_size //= 2
            # Extract more features in this depth
            for l in range(0, length - 1, 1):
                decoder["decoder-depth_"+str(d)+"-length_"+str(l)] = nn.Sequential(
                    ResBlock(
                        in_channels = in_channels,
                        out_channels = in_channels,
                        kernel_size = 3,
                        mode = "level",
                        activation_func = activation_func,
                    ),
                    nn.BatchNorm2d(in_channels),
                    activation_func(),
                    nn.Dropout(),
                )
            # Upsampling
            decoder["decoder-depth_"+str(d)+"-upsample"] = nn.Sequential(
                ResBlock(
                    in_channels = in_channels,
                    out_channels = feature_size,
                    mode = "upsample",
                    activation_func = activation_func,
                ),
                nn.BatchNorm2d(feature_size),
                activation_func(),
                nn.Dropout(),
            )
        # Last layer
        decoder["last_layer_with_no_activation"] = nn.Sequential(
            ResBlock(
                in_channels = feature_size,
                out_channels = out_channels,
                kernel_size = 3,
                mode = "level",
                activation_func = activation_func,
            ),
            CustomReLU(cut_off=zero_pixel_after_transform),
        )
        self.decoder = nn.Sequential(decoder)
        self.image_height = int(output_image_size[0])
        self.image_width = int(output_image_size[1])

    def forward(self, x):
        x = self.first_layer_to_increase_no_channels(x)
        # The output of the conv layer is (batch, feature_size, *image_size)
        # The input of the attention layer is (seq_size = image_size, batch, embed_dim = feature_size)
        x = x.flatten(start_dim=2).permute(2,0,1)
        x = self.first_positional_encoding(x) + x
        x, _ = self.first_multiheadattention(x, x, x)
        x = x.permute(1,2,0)
        x = self.activation_after_multiheadattention_and_reshape(x.view(x.size(0), x.size(1), *self.input_image_size))
        # Back to the decoder
        x = self.decoder(x)
        start_height = (x.size(-2)-self.image_height)//2
        start_width = (x.size(-1)-self.image_width)//2
        x = x[:,:,start_height:start_height+self.image_height,start_width:start_width+self.image_width]
        return x
# Function to calculate KL loss
def calculate_KL_loss(mu, logvar):
    return -0.5*(1 + logvar - mu**2 - torch.exp(logvar)).mean()

class KL_loss_vector(nn.Module):
    """
    Same as calculate_KL_loss but it is implemented as nn.Module
    It is created because I decided to move the KL into the return
    of the forward function to make sure parallel computing
    do not stack results from different GPU into the model
    You could see in the return, there is no .mean() !!!"""
    def __init__(self):
        super().__init__()
    def forward(self, mu, logvar):
        return -0.5*(1 + logvar - mu**2 - torch.exp(logvar))

# DEFINE YOUR ENCODER-DECODER ARCHITECTURE MODEL TO OUTPUT RECONSTRUCTED IMAGE AND PREDICT BRAIN AGE IN THE LATENT SPACE
# Decoder
class VAE(nn.Module):
    def __init__(self, in_channels=1, image_size=[196, 230], depth=4, length=1, encoder_complexity = 32, decoder_complexity = 32, latent_channels=32*4 , classifer_dim=[64, 1],
                 decoder_length = 3, out_channels=1, activation_func=nn.LeakyReLU, attention_dropout=0.1, zero_pixel_after_transform=0):
        """
        in_channels - integer: number of channels of input image
        image_size - tuple: size of image
        depth - integer: how many time the network downsample the image
        length - integr: how many CNN will the image underwent before downsampled,
        classifer_dim - list: dimension of hidden and last layer
        """
        assert type(activation_func) == type, f"activation_func {activation_func} should be a class"
        super().__init__()
        # Encoder layer
        self.encoder = Encoder(in_channels=in_channels, image_size=image_size, depth=depth, length=length, complexity=encoder_complexity,
                               latent_channels=latent_channels, activation_func=activation_func, attention_dropout=attention_dropout)
        self.encoder_activation = activation_func()

        number_of_linear_node = int(self.encoder.output_size.prod())
        latent_channels = int(self.encoder.output_size[0])
        # Encoder to mu and logvar
        self.mu = nn.Conv2d(
            in_channels=latent_channels,
            out_channels=latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.logvar = nn.Conv2d(
            in_channels=latent_channels,
            out_channels=latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # KL loss
        self.calculate_KL_loss_vector = KL_loss_vector()
        # Decoder layer
        self.decoder = Decoder(in_channels=latent_channels, input_image_size=self.encoder.output_size[1:3],
                               output_image_size=image_size, depth=depth, complexity=decoder_complexity,
                               length=decoder_length, out_channels=out_channels, activation_func=activation_func,
                               attention_dropout=attention_dropout, zero_pixel_after_transform=zero_pixel_after_transform)

        # Linear map to latent space
        # Classifier layer for multitask learning
        if len(classifer_dim)>1:
            self.classifier = nn.Sequential(
                MLP([number_of_linear_node, *classifer_dim[:-1]], activation_func=activation_func),
                nn.BatchNorm1d(classifer_dim[-2]),
                activation_func(),
                nn.Dropout(),
                nn.Linear(classifer_dim[-2],classifer_dim[-1]),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(number_of_linear_node,classifer_dim[-1]),
            )

        self.classification_output = 0
        self.latent_KL_loss_vector = 0
        
        # Initiate the weights
        self.initialize_weights()
        
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                                   nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                                   nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        # Encoder
        x = self.encoder_activation(self.encoder(x))
        mu = self.mu(x)
        logvar = self.logvar(x)

        # Calculate DL loss
        self.latent_KL_loss_vector = self.calculate_KL_loss_vector(mu, logvar)

        # Classifier
        self.classification_output = self.classifier(torch.flatten(mu, start_dim = 1))

        # Sampling
        if self.training:
            x = mu + torch.exp(0.5*logvar)*torch.randn_like(logvar)
        else:
            x = mu

        # Decoder
        x = self.decoder(x)

        return x, self.latent_KL_loss_vector, self.classification_output,

    def reconstruct(self, x):
        # Encoder
        x = self.encoder_activation(self.encoder(x))
        mu = self.mu(x)
        x = mu
        x = self.decoder(x)
        return x

# Define WassersteinLoss
class WassersteinLoss(nn.Module):
    """Wasserstein Loss function, implemented as:
    -ΣD(x)+ΣD(G(z)) for discriminator
    -ΣD(G(z)) for generator
    It is coded like this so it can be put into model wrapper to replace the nn.BCELoss"""
    def __init__(self):
        super().__init__()
    def forward(self, outputs, targets):
        targets=targets.mean()
        if targets==1:
            return -outputs.mean()
        elif targets==0:
            return outputs.mean()
        else:
            assert False, "targets of WassersteinLoss should be a vector of 0 or 1 only"

# Define gradient for gradient penalty
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def compute_gradient_penalty(D, real_samples, fake_samples):
    """gradient penalty to avoid the Wasserstein skyrocket to inf when trainning"""
    # Interpolate an image between the real and fake image
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # Classify the interpolated image
    d_interpolates = D(interpolates)
    # Calculate the gradient
    fake = torch.ones((real_samples.shape[0], 1), requires_grad=False).to(real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    # Calculate the gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# DEFINE YOUR DISCRIMINATOR MODEL
class Discriminator(EncoderClassifier):
    def __init__(self, in_channels=1, image_size=[196, 230],
                 depth=4, length=1, complexity = 32, last_layer_channels=32*4,
                 classifer_dim=[64, 1], activation_func=nn.LeakyReLU, attention_dropout=0.1):
        """
        in_channels - integer: number of channels of input image
        image_size - tuple: size of image
        depth - integer: how many time the network downsample the image
        length - integr: how many CNN will the image underwent before downsampled,
        classifer_dim - list: dimension of hidden and last layer
        """
        assert type(activation_func) == type, f"activation_func {activation_func} should be a class"
        assert classifer_dim[-1] == 1, "The last number in classifer_dim should be 1 for a GAN network"
        discriminator_augmented_image_size = (image_size[0], image_size[1])
        super().__init__(in_channels=in_channels, image_size=discriminator_augmented_image_size,
                         depth=depth, length=length, complexity = complexity, latent_channels=last_layer_channels,
                         classifer_dim=classifer_dim, activation_func=activation_func,
                         attention_dropout=attention_dropout)
    def forward(self, x):
        return super().forward(x)

if __name__ == '__main__':
    """
    Test function
    Run the script to test all the function
    """

    # Test each function!!!   
    """Test PositionalEncoding"""
    x = torch.rand((100, 17, 32))
    test = PositionalEncoding(embed_dim=32, dropout=0.1)
    assert test(x).size() == torch.Size([100, 17, 32]), "PositionalEncoding error"

    """Test ResBlock"""
    test = ResBlock(1, 3, 7, "level", nn.LeakyReLU)
    test = ResBlock(1, 3, 7, "upsample", Swish)
    x = torch.rand((100, 1, 31, 31))
    assert test(x).size() == torch.Size([100, 3, 62, 62]), "ResBlock error"

    """Test MLP"""
    test = MLP([3, 5, 7], Swish)
    x = torch.rand((100, 3))
    assert test(x).size() == torch.Size([100, 7]), "MLP error"

    """Test Encoder"""
    in_channels = 1
    image_size = [196, 230]
    depth = 4
    length = 1
    complexity = 32
    latent_channels= 32 #complexity*depth
    activation_func=Swish
    attention_dropout=0.1
    test = Encoder(in_channels=in_channels, image_size=image_size,
                   depth=depth, length=length, complexity=complexity, latent_channels=latent_channels,
                   activation_func=activation_func,attention_dropout=attention_dropout)
    x = torch.rand((3, 1, *image_size))
    assert test(x).size() == torch.Size([3, 32, 13, 15]), "Encoder error"

    """Test EncoderClassifier"""
    in_channels = 1
    image_size = [196, 230]
    depth = 4
    length = 2
    complexity = 16
    latent_channels=64 #latent_channels
    classifer_dim=[64,64,1]
    activation_func=Swish
    attention_dropout=0.3
    test = EncoderClassifier(in_channels=in_channels, image_size=image_size, depth=depth, length=length, complexity=complexity,
                             latent_channels=latent_channels, classifer_dim=classifer_dim, activation_func=activation_func,
                             attention_dropout=attention_dropout)
    x = torch.rand((3, 1, *image_size))
    assert test(x).size() == torch.Size([3, 1]), "EncoderClassifier error"

    """Test CustomReLU"""
    test = CustomReLU(-2)
    x = torch.tensor([[1, 1.0, -2, -3, -4],[1, -3, 2, 1, 1]])
    assert (test(x) == torch.tensor([[1,1,-2,-2,-2],[1,-2,2,1,1]])).all(), "CustomReLU error"

    """Test MaskedMSELoss"""
    x = torch.tensor([[1, 1.0, 1, 1, 1],[1, 1, 1, 1, 1]])
    y = torch.tensor([[0, 0, 0, 0, 0],[-2, -2, 0, 0.0, 0]])
    test=MaskedMSELoss(cut_off = -1, non_zero_weight = 0.5, zero_weight = 0.5)
    assert test(x,y) == 5, "MaskedMSELoss error"

    """Test Decoder"""
    in_channels=4
    input_image_size = [25, 29]
    output_image_size = [196, 230]
    depth = 3
    length = 1
    complexity= 32
    out_channels = 1
    attention_dropout=0.3
    zero_pixel_after_transform=0
    test = Decoder(in_channels=in_channels, input_image_size=input_image_size, output_image_size=output_image_size,
                   depth=depth, length=length, complexity=complexity, out_channels=out_channels, activation_func=Swish,
                   attention_dropout=attention_dropout, zero_pixel_after_transform=zero_pixel_after_transform)

    x = torch.rand((3, in_channels, *input_image_size))
    assert test(x).size() == torch.Size([3, 1, 196, 230]), "Decoder error"

    """Test VAE"""
    in_channels = 1
    image_size = [196, 230]
    depth = 3
    length = 1
    encoder_complexity = 32
    decoder_complexity = 32
    latent_channels = 4
    classifer_dim=[64, 64, 1]
    out_channels=1
    decoder_length=1
    activation_func=Swish
    attention_dropout=0.3
    zero_pixel_after_transform=0
    test = VAE(in_channels=in_channels, image_size=image_size, depth=depth, length=length, encoder_complexity=encoder_complexity, decoder_complexity=decoder_complexity,
            latent_channels=latent_channels, classifer_dim=classifer_dim, out_channels=out_channels, decoder_length=decoder_length, activation_func=activation_func,
            attention_dropout=attention_dropout, zero_pixel_after_transform=zero_pixel_after_transform)
    x = torch.rand((3, 1, *image_size))
    assert test(x)[0].size() == torch.Size([3, 1, 196, 230]), "VAE error"
    assert test(x)[1].size() == torch.Size([3, 4, 25, 29]), "VAE error"
    assert test(x)[2].size() == torch.Size([3, 1]), "VAE error"

    """Test WassersteinLoss"""
    test = WassersteinLoss()
    ones = torch.ones(4)
    zeros = torch.zeros(4)
    x = torch.ones(4)*5
    assert test(x,ones) == -5, "WassersteinLoss error"
    assert test(x,zeros) == 5, "WassersteinLoss error"

    """Test Discriminator"""
    in_channels = 1
    image_size = [196, 230]
    discriminator_augmented_image_size = (image_size[0]*3//4, image_size[1]*3//4)
    depth = 5
    length = 2
    complexity = 32
    last_layer_channels=complexity*depth #latent_channels
    classifer_dim=[64,64,1]
    activation_func=Swish
    attention_dropout=0.1
    test = Discriminator(in_channels=in_channels, image_size=discriminator_augmented_image_size, depth=depth, length=length,
                        complexity=complexity, last_layer_channels=last_layer_channels, classifer_dim=classifer_dim,
                        activation_func=activation_func, attention_dropout=attention_dropout)

    x = torch.rand((3, 1, *discriminator_augmented_image_size))
    assert test(x).size() == torch.Size([3, 1]), "Discriminator error"

    # End the tests
    print("All tests passed")
