# utils/data_loader.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size=128, is_bernoulli=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.bernoulli(x) if is_bernoulli else x)
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    return DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
