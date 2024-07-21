# VAE-GAN from scratch to Generate Neonatal Brain Images and Estimate Brain Age

This repository is my coursework for Advanced Machine Learning at King's College London. It consists of creating a VAE-GAN network to meet some predefined target set by the lecturer.

## Aims:

Creating a VAE-GAN network from scratch for generating neonatal brain images and estimate brain age from the features in the latents space.

## Objectives:

+ Mean absolute error of age prediction: < 1 week

+ Mean square error of reconstruction of VAE network: < 0.01

+ Visual inference of the generated images from the Decoder of VAE-GAN network

## Method:

There are three main networks in this projects

+ **Age Regression network**: I implemented an **Encoder**-like structure, with **Residual Blocks** as building blocks to downsample an image from (1, 196, 230) to (64, 13, 15). I then project them to a **Multilayer Perceptron** to predict the age.

+ **VAE network**: The **Encoder** is similar to the **Age Regression network** without the last linear layers. **Convolution Layer** is applied after the **Encoder** to get the probability distribution of the latent images (for Kullback–Leibler divergence (KL) loss). The **Decoder** mirrors the structures of the **Encoder**, upsamples the latent images and crops them back to original size. Additionally, I added a **Multi-Head Attention** layer for the decoder to introduce some spatial invariance for the model. A **Multilayer Perceptron** is used to predict the age using the features extracted in the latent space. The network is trained as a **multi-task network**, with a weighted sum of 3 losses:

    + Reconstruction loss: to reconstruct the images

    + KL loss: to force the latent space to follow a normal distribution, and make the network learn more meaningful features

    + Regression loss: to predict the age of the images

+ **VAE-WGAN network**: the **Encoder** of the **VAE network** acts as a **Generator**. The **Discriminator** has the same structure as **Age Regression network**, where the outputs of the linear layer return the probabilities of the images being real or fake (instead of age). **Wasserstein loss** is used to mitigate the gradient vanishing and **Gradient Penalty** is used to avoid the **Discriminator** outputs go to infinity. The network is trained as a **multi-task network**, with a weighted sum of 4 losses for **generator** and 2 losses for **discriminator**:

    + **Generator** loss: 3 losses from **VAE network** above together with **Wasserstein loss** 

    + **Discriminator** loss: **Wasserstein loss** with **Gradient Penalty**

+ Other details:

    + I used **Swish** activation function instead of **LeakyReLU**. [(Reference)](https://arxiv.org/abs/1710.05941).

    + I used **SmoothL1Loss** instead of **L1Loss** for age regression. [(Reference)](https://someshfengde.medium.com/understanding-l1-and-smoothl1loss-f5af0f801c71).

    + The order of regularisation in all structures are **BatchNorm** -> **Activation** -> **DropOut**
 
## Result:
I achieved the target loss proposed by the lecturers, and successfully generated the required images:

+ For the age regression, the network could predict the fetal age with a mean error of 0.8 week.

+ For the VAE network, I achieved a reconstruction loss of 0.001, which is 10 times lower then what is required by the lecturer

+ For the VAE-WGAN, adding a discriminator reduced the number of epochs I needed for training the VEA, however, the generated images' visual qualities didn't improve much.

## Some comments:

Age regression:

+ SmoothL1Loss worked better than L1Loss for age regression. As when I was training with L1Loss, the MAE got stuck and fluctate with no improved sign, the loss sometime skyrocket (which might be due to the that that L1 Loss is non-differential at zeros).

+ Swish activation help smoothen the loss compared to LeakyReLU.

+ Loss in trainning is higher than validation because of the dropout layer, which force the model to learn multiple representation of the images. And when no dropouts is active, the model is an ensemble model of smaller models with dropouts.

VAE

+ At first, I tried a fully convolutional structure for the decoder, however, it didn't work well:

![loss when trained with a FCN](image/Losses_of_network_with_all_conv.png)
![result when trained with a FCN](image/Results_of_network_with_all_conv.png)

+ This could be explain by the fact that convolutional layers are spatial invariance and could not learn the spatial characteristics. When we look at the KL, it is increased because the decoder could not learn a latent representation that is normally distributed (It need to learn a latent representation that maintains the spatial structures). Thereforem I decided to add a **Multi-Head Attention** layer for the decoder, so the decoder could reconstruct the image from a normally distributed latent image. And IT WORKED MUCH BETTER!!!

![loss when attention layer added](image/Losses_of_network_with_attention_added.png)
![result when attention layer added](image/Results_of_network_with_attention.png)

+ Trainning of the network is really tricky, as the background accounts for >50% of the pixels. I decided to add a mask and weight the background and foreground differently. Additionally, at the final layer, instead of using a tanh or sigmoid function, I used a ReLU with a custom cut-off that maps every pixels smaller than the background become the background! This change made the VEA results much better!!!

VAE-WGAN

+ The GAN network is really hard to train, I really struggled with finding a ballance between the generator and discriminator.

+ The discriminator easily overfitted the data, therefore, I added an augmentation for it specifically. The augmentations includes random rotate, tranlate, shearing, flipping, and cropping. My premise is that, a real image is still a real image after these transformation.

+ I decided not to spend my time further, because I only have 700 images (I believe the number of images for GAN can easily go up to thousands or millions).