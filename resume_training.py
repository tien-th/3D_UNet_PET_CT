import sys
import os
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from utils import get_image_grid

experiment_name = "Experiment_ResidualUNet3D_lr1e-4"
N_EPOCHS = 100
LOG_DIR = f'./logs/{experiment_name}'
CHECKPOINT_DIR = LOG_DIR + '/checkpoints'
CHECKPOINT_PATH = CHECKPOINT_DIR + '/model_epoch_13.pth'  # Update to the latest checkpoint if needed

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

from model.unet3d import ResidualUNet3D, ResidualUNetSE3D
from dataset.dataset import CustomAlignedDataset 

# Dataset and dataloader
root_path = '3d_model_data'
train_dataset = CustomAlignedDataset(root_path, stage='train')
val_dataset = CustomAlignedDataset(root_path, stage='val')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize model
IN_CHANNELS, OUT_CHANNELS = 1, 1
model = ResidualUNetSE3D(IN_CHANNELS, OUT_CHANNELS).to('cuda')

# Optimizer and Scheduler
learning_rate = 1e-4
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # Adjust step_size and gamma as needed

# TensorBoard writer
writer = SummaryWriter(LOG_DIR)

# Load checkpoint if available
start_epoch = 0
if os.path.isfile(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']  # Resume from the saved epoch
    print(f"Resuming from epoch {start_epoch}")

# Training loop
for epoch in range(start_epoch, N_EPOCHS):
    model.train()
    train_loss = 0.0

    for i, (target, data) in tqdm(enumerate(train_loader)):
        data, target = data.to('cuda').float(), target.to('cuda').float()
        optimizer.zero_grad()
        output = model(data)

        # Loss calculation
        loss = (output - target).abs().mean()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{N_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            writer.add_scalar('Training Loss/step', loss.item(), epoch * len(train_loader) + i)

        if i % 1000 == 0 :
            # Select first 4 slices along the depth dimension (assume data is in shape [B, C, D, H, W])
            input_slices = get_image_grid(data[0, 0, :4, :, :].cpu().unsqueeze(1))
            output_slices = get_image_grid(output[0, 0, :4, :, :].cpu().unsqueeze(1))
            target_slices = get_image_grid(target[0, 0, :4, :, :].cpu().unsqueeze(1))
            
            # Log images to TensorBoard as image grids
            writer.add_image(f'epoch_{epoch+1}/{i}_Input_slices_train', input_slices, epoch + 1, dataformats='HWC')
            writer.add_image(f'epoch_{epoch+1}/{i}_Output_slices_train', output_slices, epoch + 1, dataformats='HWC')
            writer.add_image(f'epoch_{epoch+1}/{i}_GroundTruth_slices_train', target_slices, epoch + 1, dataformats='HWC')

    # Log the average loss for the epoch
    avg_train_loss = train_loss / len(train_loader)
    writer.add_scalar('Training Loss/epoch', avg_train_loss, epoch + 1)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data, target = data.to('cuda').float(), target.to('cuda')
            output = model(data)
            val_loss += (output - target).abs().mean().item()
            
            # Log images for one random voxel in the first batch only
            if idx == 0:
                # Select first 4 slices along the depth dimension (assume data is in shape [B, C, D, H, W])
                input_slices = get_image_grid(data[0, 0, :4, :, :].cpu().unsqueeze(1))
                output_slices = get_image_grid(output[0, 0, :4, :, :].cpu().unsqueeze(1))
                target_slices = get_image_grid(target[0, 0, :4, :, :].cpu().unsqueeze(1)) 
                
                # Log images to TensorBoard as image grids
                writer.add_image(f'epoch_{epoch+1}/valid_{i}_Input_slices', input_slices, epoch + 1, dataformats='HWC')
                writer.add_image(f'epoch_{epoch+1}/valid_{i}_Output_slices', output_slices, epoch + 1, dataformats='HWC')
                writer.add_image(f'epoch_{epoch+1}/valid_{i}_GroundTruth_slices', target_slices, epoch + 1, dataformats='HWC')
    
    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar('Validation Loss/epoch', avg_val_loss, epoch + 1)
    print(f"\nEpoch [{epoch+1}/{N_EPOCHS}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    # Step the scheduler
    scheduler.step()
    
    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Close the writer
writer.close()