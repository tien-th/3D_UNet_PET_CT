import sys
import os
import torch
from tqdm import tqdm

experiment_name = "Experiment_ResidualUNet3D_lr1e-4"
N_EPOCHS = 100
LOG_DIR = f'./logs/{experiment_name}'
CHECKPOINT_DIR = LOG_DIR + '/checkpoints'
CHECKPOINT_PATH = CHECKPOINT_DIR + '/model_epoch_65.pth'  # Update to the latest checkpoint if needed

from model.unet3d import ResidualUNet3D, ResidualUNetSE3D

# Initialize model
IN_CHANNELS, OUT_CHANNELS = 1, 1
model = ResidualUNetSE3D(IN_CHANNELS, OUT_CHANNELS).to('cuda')
# read 3D data 
import numpy as np
import torch

# Load checkpoint if available
if os.path.isfile(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']  # Resume from the saved epoch
    print(f"Resuming from epoch {start_epoch}")

model = model.to('cuda')

txt_list_patients = '3d_model_data/test.txt'
result_path = '/workdir/radish/PET-CT/3D_reggan/metric'
with open(txt_list_patients) as f:
    lines = f.readlines()
    patient_list = [x.strip() for x in lines]

for patient_folder in tqdm(patient_list):
    file_path = patient_folder + '/predicted_volume.npy'
    voxel = np.load(file_path, allow_pickle=True)
    # Define the depth of each chunk and overlap
    CHUNK_DEPTH = 7
    OVERLAP = 5  # Overlapping depth for each chunk
    D, H, W = voxel.shape  # Assuming voxel shape is (D, 256, 256)

    # Ensure voxel is on GPU
    voxel = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to('cuda')  # Shape: (1, 1, D, 256, 256)
    voxel = voxel / 32767.0 
    voxel =  (voxel - 0.5 ) * 2 

    # Prepare tensors to store accumulated output and weights
    output_accum = torch.zeros((1, 1, D, H, W), dtype=torch.float32).to('cuda')  # Accumulated output
    weight_accum = torch.zeros((1, 1, D, H, W), dtype=torch.float32).to('cuda')  # Weight mask

    # Process the voxel in chunks along the depth dimension
    start_idx = 0
    while start_idx < D:
        # print(f"Processing chunk starting at index {start_idx}...")
        # Calculate end index for the chunk, with overlap
        end_idx = min(start_idx + CHUNK_DEPTH, D)
        
        # If this is the last chunk and smaller than CHUNK_DEPTH, adjust start_idx for overlap
        tmp = start_idx
        if end_idx - start_idx < CHUNK_DEPTH and start_idx != 0:
            start_idx = max(0, end_idx - CHUNK_DEPTH)  # Adjust start_idx to include overlap with the previous chunk
        
        # Select the chunk from the voxel
        chunk = voxel[:, :, start_idx:end_idx, :, :]  # Shape: (1, 1, CHUNK_DEPTH, 256, 256) or smaller if at the last chunk

        # Pass the chunk through the model
        with torch.no_grad():
            output_chunk = model(chunk)

        # Determine the slice in the full output corresponding to this chunk
        output_start = start_idx
        output_end = min(start_idx + CHUNK_DEPTH, D)
        
        # Add the model's output to the accumulated tensor
        output_accum[:, :, output_start:output_end, :, :] += output_chunk[:, :, :output_end - output_start, :, :]
        weight_accum[:, :, output_start:output_end, :, :] += 1  # Increment weights for overlapping regions

        # Move to the next chunk position, taking the overlap into account
        # print(start_idx, CHUNK_DEPTH, OVERLAP)
        start_idx = tmp
        start_idx = start_idx + CHUNK_DEPTH - OVERLAP
        # print(start_idx)

    # Normalize the accumulated output by the weight mask to compute the average for overlapping regions
    output_voxel = output_accum / weight_accum
    # output_voxel = output_voxel.squeeze().cpu().numpy()  # Convert back to numpy if needed
    output_voxel = output_voxel.mul(0.5).add(0.5).clamp(0, 1)
    output_voxel = output_voxel * 32767.0
    output_voxel = output_voxel.squeeze().cpu().numpy()

    patient_result_path = os.path.join(result_path, os.path.basename(patient_folder))
    os.makedirs(patient_result_path, exist_ok=True)
    output_file_path = os.path.join(patient_result_path, f'3D_model_overlap_{OVERLAP}.npy')
    np.save(output_file_path, output_voxel)
    print(f"Saved predicted volume to {output_file_path}")
    # np.save('/workdir/radish/PET-CT/3D_reggan/dunganh/1.2.840.113619.2.55.3.663376.78.1704246084.428/3D_model_overlap.npy', output_voxel)