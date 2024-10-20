import numpy as np
import torch
import segmentation_models_pytorch
from tqdm import tqdm

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            labels[labels == 255] = 3  # Assuming 255 should be mapped to 3
            labels = torch.clamp(labels, 0, 3)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader):
        labels[labels == 255] = 3  # Assuming 255 should be mapped to 3
        labels = torch.clamp(labels, 0, 3)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
    
def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)
