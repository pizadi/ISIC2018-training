# ISIC2018-training
This repository contains the pytorch implementation of a number of ML models designed for segmentation of the ISIC-2018 dataset.

## U-Net
A simplified implementation of the U-Net encoder-decoder network.

## DoubleU-Net
A U-net-like network with an encoder-decoder-encoder-decoder architecture, using a pre-trained VGG19 network as its first encoder. Both of the decoder outputs are separately trained to improve the accuracy of the network.

## U-Net-Res50
A U-net-like network, using a pre-trained ResNet50 as its encoder.

## BCDU-Net
An encoder-decoder network, using bi-directional LSTMs in its decoder blocks
