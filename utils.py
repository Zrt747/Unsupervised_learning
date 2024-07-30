import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def random_crop(image, crop_size):
    width, height = image.size
    crop_width, crop_height = crop_size
    x = random.randint(0, width - crop_width)
    y = random.randint(0, height - crop_height)
    return image.crop((x, y, x + crop_width, y + crop_height))

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)

class ContrastiveModel(nn.Module):
    def __init__(self, encoder, projection_dim):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder
        self.projection_head = ProjectionHead(self.encoder.fc.in_features, projection_dim)
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return projections

def contrastive_loss(projections1, projections2, temperature=0.5):
    batch_size = projections1.size(0)
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    projections = torch.cat([projections1, projections2], dim=0)
    similarity_matrix = torch.matmul(projections, projections.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(labels.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(labels.device)
    return nn.CrossEntropyLoss()(logits / temperature, labels)