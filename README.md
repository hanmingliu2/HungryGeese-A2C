# HungryGeese-A2C
A PyTorch implementation of the A2C algorithm for the HungryGeese competition

## Recommended Readings
1. [HungryGeese competition](https://www.kaggle.com/c/hungry-geese)
2. [A3C algorithm paper](https://arxiv.org/abs/1602.01783v2)

## Architecture
1. Both the actor and the critic **share the same feature extractor** which is a 10-block ResNet with 64 filters in each block.
2. An average pooling layer is added to the end of the feature extractor to reduce model size.
3. Action masking is used to prevent the agent from performing invalid actions.
4. The Adam optimizer is used instead of RMSProp (paper).

## Note
The implementation is designed to be used with an Nvidia GPU of at least 6GB of video memory. Please feel free to modify the code based on your hardware specification.

## Attributions
1. The implementation is heavily based on [this blog post](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f) by Chris Yoon.
2. The action masking mechanism is inspired by [this paper](https://arxiv.org/abs/2006.14171) by Shengyi Huang and Santiago Ontañón.
