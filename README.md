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

+ **VAE network**: The **Encoder** is similar to the **Age Regression network** without the last linear layers. **Convolution Layer** is applied after the **Encoder** to get the probability distribution of the latent images (for Kullback–Leibler divergence (KL) loss). The **Decoder** mirrors the structures of the **Encoder**, upsamples the latent images and crops them back to original size. A **Multilayer Perceptron** is used to predict the age using the features extracted in the latent space. The network is trained as a **multi-task network**, with a weighted sum of 3 losses:

    + Reconstruction loss: to reconstruct the images

    + KL loss: to force the latent space to follow a normal distribution, and make the network learn more meaningful features

    + Regression loss: to predict the age of the images

+ **VAE-WGAN network**: the **Encoder** of the **VAE network** acts as a **Generator**. The **Discriminator** has the same structure as **Age Regression network**, where the outputs of the linear layer return the probabilities of the images being real or fake (instead of age). **Wasserstein loss** is used to mitigate the gradient vanishing and **Gradient Penalty** is used to avoid the **Discriminator** outputs go to infinity. The network is trained as a **multi-task network**, with a weighted sum of 4 losses for **generator** and 2 losses for **discriminator**:

    + **Generator** loss: 3 losses from **VAE network** above together with **Wasserstein loss** 

    + **Discriminator** loss: **Wasserstein loss** with **Gradient Penalty**

+ Other details:

    + I used **Swish** activation function instead of **LeakyReLU**. [(Reference)](https://arxiv.org/abs/1710.05941).

    + I used **SmoothL1Loss** instead of **L1Loss** or **L2Loss**. [(Reference)](https://someshfengde.medium.com/understanding-l1-and-smoothl1loss-f5af0f801c71).

    + The order of regularisation in all structures are **BatchNorm** -> **Activation** -> **DropOut**

## Result:

I achieved the target loss proposed by the lecturers, and successfully generated the required images. The loss curves showed the potentials for my model to improve with more training epochs (I decided not to spend more of my time anyway). As I was only auditing this course, no official grade was given.