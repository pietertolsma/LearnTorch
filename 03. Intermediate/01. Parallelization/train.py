
# Train in parallel

import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# import tensorboard]
from torch.utils.tensorboard import SummaryWriter

from models.simplenet import SimpleNet

# Set the seed for reproducibility
torch.manual_seed(0)

# Define the device
device = "cpu"

# Define the hyperparameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Define the transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download the training dataset
train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12321'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, model):
    setup(rank, world_size)
    print(rank)

    # Distributed data
    train_sampler = data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=1,
        rank=0
    )
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )

    writer = None
    if rank == 0:
        writer = SummaryWriter(f'runs/Parallelization_11')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(int(5)):
        
        loss_list = []
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            # Write loss to tensorboard
        if rank == 0:
            print(sum(loss_list)/len(loss_list))
            writer.add_scalar('training loss', sum(loss_list)/len(loss_list), epoch)

    cleanup()



if __name__ == "__main__":
    world_size = 4
    model = SimpleNet()
    model.share_memory()
    # NOTE: this is required for the ``fork`` method to work
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=train, args=(rank, world_size, model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()