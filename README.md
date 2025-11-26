Diffusion model for mobility trajectories. A lightweight custom-designed deep learning model in Pytorch with an efficient training schedule.

The goal is to use the parallelism of Diffusion model to generate GPS locations in vector space that not only accurately represent the historical sequence of movements but also generate valid trajectories that are on the streets.

This repository consists of several versions of models that gradually develop into tile-based diffusion models so that each tile contains a diffusion model that hands over its generated trajectory to the next diffusion model. The score function of the diffusion models consists of a deep learning model that learns the shape of the trajectories and their distances to the street points independently and finally merges them together. Each section of the model may use various techniques for learning. We used a dilated convolutional neural network, but it can be replaced with LSTM or transformer encoder, etc.

The model is named by "V#" and the main files are named as "main_V#". The version number on main files refers to the minimum model version that it is compatible with. The maximum model version is determined by the next main file with higher version number.
