import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model.unet3d import ResidualUNetSE3D
from dataset.dataset import CustomAlignedDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Set the master address
    os.environ['MASTER_PORT'] = '29700'     # Set a free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Adjust device if necessary
    state_dict = checkpoint['model_state_dict']

    # Check if `module.` prefix exists in the DDP model keys
    is_ddp = any(key.startswith("module.") for key in model.state_dict().keys())
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if is_ddp and not k.startswith("module."):
            # Add 'module.' prefix for single-GPU checkpoint to match DDP model
            new_state_dict[f"module.{k}"] = v
        elif not is_ddp and k.startswith("module."):
            # Remove 'module.' prefix if loading DDP checkpoint into non-DDP model
            new_state_dict[k[7:]] = v
        else:
            # No prefix adjustment needed
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return checkpoint

def main(rank, world_size):
    setup(rank, world_size)
    experiment_name = "Experiment_3_slices"
    N_EPOCHS = 100
    LOG_DIR = f'./logs/{experiment_name}'
    CHECKPOINT_DIR = LOG_DIR + '/checkpoints'
    # CHECKPOINT_PATH = CHECKPOINT_DIR + '/model_epoch_38.pth'
    CHECKPOINT_PATH = ''

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Dataset and Dataloader
    root_path = '3d_model_data_3'
    train_dataset = CustomAlignedDataset(root_path, stage='train')
    val_dataset = CustomAlignedDataset(root_path, stage='val')
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler)
    
    # Model
    IN_CHANNELS, OUT_CHANNELS = 1, 1
    model = ResidualUNetSE3D(IN_CHANNELS, OUT_CHANNELS).to(rank)
    model = DDP(model, device_ids=[rank])

    # Optimizer and Scheduler
    learning_rate = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Load checkpoint if available
    start_epoch = 0
    if os.path.isfile(CHECKPOINT_PATH):
        if rank == 0: print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = load_checkpoint(model, CHECKPOINT_PATH)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        if rank == 0: print(f"Resuming from epoch {start_epoch}")

    # TensorBoard writer (only on rank 0)
    if rank == 0:
        writer = SummaryWriter(LOG_DIR)

    # Training loop
    for epoch in range(start_epoch, N_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        
        for i, (target, data) in tqdm(enumerate(train_loader), disable=rank != 0):
            data, target = data.to(rank).float(), target.to(rank).float()
            optimizer.zero_grad()
            output = model(data)

            # Loss calculation
            loss = (output - target).abs().mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if rank == 0 and i % 10 == 0:
                print(f"Epoch [{epoch+1}/{N_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                writer.add_scalar('Training Loss/step', loss.item(), epoch * len(train_loader) + i)

        # Reduce training loss across all processes
        train_loss_tensor = torch.tensor(train_loss, device=rank)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = train_loss_tensor.item() / (len(train_loader) * world_size)

        if rank == 0:
            writer.add_scalar('Training Loss/epoch', avg_train_loss, epoch + 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx, (data, target) in enumerate(val_loader):
                data, target = data.to(rank).float(), target.to(rank)
                output = model(data)
                val_loss += (output - target).abs().mean().item()

        # Reduce validation loss across all processes
        val_loss_tensor = torch.tensor(val_loss, device=rank)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)

        if rank == 0:
            writer.add_scalar('Validation Loss/epoch', avg_val_loss, epoch + 1)
            print(f"\nEpoch [{epoch+1}/{N_EPOCHS}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Step the scheduler
        scheduler.step()

        # Save checkpoint (only on rank 0)
        if rank == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),  # Save without 'module.'
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # Close the writer
    if rank == 0:
        writer.close()

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)