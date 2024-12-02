import sys
import os
import torch
from tqdm import tqdm
import numpy as np

experiment_name = "Experiment_ResidualUNet3D_lr1e-4"
N_EPOCHS = 100
LOG_DIR = f'./logs/{experiment_name}'
CHECKPOINT_DIR = LOG_DIR + '/checkpoints'
CHECKPOINT_PATH = CHECKPOINT_DIR + '/model_epoch_65.pth'  # Update to the latest checkpoint if needed

from model.unet3d import ResidualUNet3D, ResidualUNetSE3D

# Initialize model
IN_CHANNELS, OUT_CHANNELS = 1, 1
model = ResidualUNetSE3D(IN_CHANNELS, OUT_CHANNELS).to('cuda')



# Load checkpoint if available
if os.path.isfile(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']  # Resume from the saved epoch
    print(f"Resuming from epoch {start_epoch}")

model = model.to('cuda')


# read 3D data 

txt_list_patients = '3d_model_data/test.txt'
result_path = '/workdir/radish/PET-CT/3D_reggan/metric'
with open(txt_list_patients) as f:
    lines = f.readlines()
    patient_list = [x.strip() for x in lines]

for patient_folder in tqdm(patient_list):
    # file_path = '/workdir/radish/PET-CT/3D_reggan/dunganh/1.2.840.113619.2.55.3.663376.78.1704246084.428/predicted_volume.npy'
    file_path = patient_folder + '/predicted_volume.npy'
    voxel = np.load(file_path, allow_pickle=True)

    # Define the depth of each chunk and overlap
    CHUNK_DEPTH = 7
    D, H, W = voxel.shape  # Assuming voxel shape is (D, 256, 256)

    # Ensure voxel is on GPU
    voxel = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to('cuda')  # Shape: (1, 1, D, 256, 256)
    voxel = voxel / 32767.0 
    voxel =  (voxel - 0.5 ) * 2 
    # Prepare an empty list to store output chunks
    output_chunks = []

    # Process the voxel in chunks along the depth dimension
    start_idx = 0
    while start_idx < D:
        # Calculate end index for the chunk, with overlap
        end_idx = min(start_idx + CHUNK_DEPTH, D)
        
        # If this is the last chunk and smaller than CHUNK_DEPTH, overlap with the previous chunk
        if end_idx - start_idx < CHUNK_DEPTH and start_idx != 0:
            start_idx = max(0, end_idx - CHUNK_DEPTH)  # Adjust start_idx to include overlap with the previous chunk
        
        
        # Select the chunk from the voxel
        chunk = voxel[:, :, start_idx:end_idx, :, :]  # Shape: (1, 1, CHUNK_DEPTH, 256, 256) or smaller if at the last chunk

        # Pass the chunk through the model
        with torch.no_grad():
            output_chunk = model(chunk)

        # Adjust output chunk size to match only the new portion without redundant overlaps
        # output_chunk_trimmed = output_chunk[:, :, start_idx % CHUNK_DEPTH:, :, :] if start_idx != 0 else output_chunk
        output_chunk_trimmed = output_chunk if start_idx % CHUNK_DEPTH == 0 else output_chunk[:, :, -(D % CHUNK_DEPTH):, :, :]
        if end_idx == D:
            print("End of voxel reached", output_chunk_trimmed.shape, start_idx, end_idx)
        output_chunks.append(output_chunk_trimmed)

        # Move to the next chunk position, taking the overlap into account
        start_idx += CHUNK_DEPTH

    # Concatenate all processed chunks along the depth dimension
    output_voxel = torch.cat(output_chunks, dim=2)  # Shape: (1, 1, D, 256, 256)

    # print(output_voxel.shape, output_voxel.min(), output_voxel.max())

    # Remove extra dimensions to match (D, 256, 256)
    output_voxel = output_voxel.mul(0.5).add(0.5).clamp(0, 1)
    output_voxel = output_voxel * 32767.0
    output_voxel = output_voxel.squeeze().cpu().numpy() 

    # print(output_voxel.shape, voxel.shape, output_voxel.min(), output_voxel.max())
    # Save the output voxel

    patient_result_path = os.path.join(result_path, os.path.basename(patient_folder))
    os.makedirs(patient_result_path, exist_ok=True)

    output_file_path = os.path.join(patient_result_path, '3D_model_non_overlap.npy')
    np.save(output_file_path, output_voxel)
    print(f"Saved predicted volume to {output_file_path}")

    # np.save('/workdir/radish/PET-CT/3D_reggan/dunganh/1.2.840.113619.2.55.3.663376.78.1704246084.428/3D_model_non_overlap.npy', output_voxel)