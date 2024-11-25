# import sys
# import os
# import torch

# import debugpy 

# sys.path.append("..")
# from Unet3d import ResidualUNet3D, ResidualUNetSE3D
# # from diffusers.models.unets.custom_unet_3d_condition import UNet3DConditionModel

# # debugpy.listen(5678)
# # print("Waiting for debugger attach")
# # debugpy.wait_for_client()
# # print("Debugger attached")

# SAMPLE_SIZE = (256, 256)
# IN_CHANNELS = 1
# OUT_CHANNELS = 1

# model = ResidualUNetSE3D(IN_CHANNELS, IN_CHANNELS) 

# # count_parameters(model) 
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())

# print(count_parameters(model))
# input = torch.randn(1, 1, 6, 256, 256)

# input = input.to('cuda:1')
# model = model.to('cuda:1')

# print(model)

# output = model(input)
# print(output.shape)


import sys
import os
import torch

# Uncomment these if debugging is needed
# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# print("Debugger attached")

sys.path.append("..")
from model.Unet3d import ResidualUNet3D, ResidualUNetSE3D

# Constants
SAMPLE_SIZE = (256, 256)
IN_CHANNELS = 1
OUT_CHANNELS = 1

# Initialize model and count parameters
model = ResidualUNetSE3D(IN_CHANNELS, OUT_CHANNELS)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
print(f"Model Parameters: {count_parameters(model)}")

# Set up input and move to GPU
input_tensor = torch.randn(1, 1, 7, 256, 256).to('cuda:1')
model = model.to('cuda:1')

# Forward pass
output = model(input_tensor)
print(f"Output Shape: {output.shape}")

# Dummy target and loss for testing backward pass
target = torch.randn_like(output).to('cuda:1')
loss_fn = torch.nn.MSELoss()  # Mean squared error for simplicity
loss = loss_fn(output, target)

# Backward pass
loss.backward()

print("Backward pass completed successfully.")