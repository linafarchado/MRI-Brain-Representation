import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch
from trainAndTest import train, evaluate, test
from torchvision import transforms
import matplotlib.pyplot as plt

import sys
import os

# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Dataloader import CustomDatasetWithLabels, CustomDataset, CustomDatasetWithLabelsFiltered
from Utils import EarlyStopping
from Metrics import calculate_dice_per_class
from Visualize import visualize_segmentation, visualize_segmentation_Dice

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.model = segmentation_models_pytorch.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=out_channels
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)

class Pipeline():
    def __init__(self, model, dataset, visualize, outputs, load, batch_size=8, start_epochs=0, total_epochs=50, has_labels=True):
        self.outputs = outputs
        self.visualize = visualize
        os.makedirs(self.outputs, exist_ok=True)
        self.has_labels = has_labels
        self.load = load if len(load) > 0 else outputs
        self.dataset = CustomDatasetWithLabelsFiltered(dataset)
        #self.num_classes = self.dataset_before_filter.get_num_classes()
        self.train_size = int(0.8 * len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size])
        self.test_dataset = CustomDatasetWithLabelsFiltered(dataset, is_training=False) if self.has_labels else CustomDataset(dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.best_loss = float('inf')
        self.best_model = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.writer = SummaryWriter(f'runs/{outputs}')
        self.total_epochs = total_epochs
        self.start_epochs = start_epochs
        self.early_stopping = EarlyStopping(patience=5, verbose=True)
        self.data_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=5),
        ])

    def save_best_model(self, epoch, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model = self.model.state_dict()
            model_path = os.path.join(self.outputs, f'best_model_epoch_{epoch}.pth')
            torch.save(self.best_model, model_path)

    def train(self):
        # Training loop
        for epoch in range(self.start_epochs, self.total_epochs):
            train_loss = train(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
            val_loss = evaluate(self.model, self.val_loader, self.criterion, self.device)
            
            print(f'Epoch {epoch+1}/{self.total_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"Early stopping, epoch: {epoch}")
                break

            self.save_best_model(epoch, val_loss)

        self.writer.close()
    
        # Save the trained model
        torch.save(self.model.state_dict(), f'{self.outputs}.pth')

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f'{self.load}.pth')))
        if self.has_labels:
            # Test the model
            preds, labels = test(self.model, self.test_loader, self.device)

            # Calculate accuracy
            accuracy = np.mean(preds == labels)
            print(f'Accuracy: {accuracy:.4f}')

            # Visualize 50 random samples
            for i in range(50):
                random_idx = np.random.randint(len(self.test_dataset))
                test_image, test_label = self.test_dataset[random_idx]
                test_image = test_image.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    test_output = self.model(test_image)
                    test_pred = torch.argmax(test_output, dim=1).squeeze().cpu().numpy()
                    dice_pred = calculate_dice_per_class(torch.from_numpy(test_pred).long(), test_label, 4)
                    patient_idx = self.test_dataset.get_patient_idx(random_idx) + 9
                visualize_segmentation_Dice(test_image.cpu().numpy(), test_label.numpy(), test_pred, f'seg_{random_idx+1}',  dice_pred, patient_idx, folder=self.visualize)

        else:
            self.visualize_predictions(self.visualize)
    
    def inference(self):
        self.model.eval()
        predictions = []
        images = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                images.extend(batch.cpu().numpy())
        
        return images, predictions

    def visualize_predictions(self, folder, num_samples=100):
        images, predictions = self.inference()
        
        for i in range(min(num_samples, len(images))):
            image = images[i]
            pred = predictions[i]
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(pred, cmap='viridis')
            plt.title('Predicted Segmentation')
            plt.axis('off')
            
            plt.savefig(os.path.join(folder, f'prediction_{i+1}.png'))
            plt.close()

def main(dataset, outputs, visualize, train=True, load="",  total_epochs=50, start_epochs=0, batch_size=8, has_labels=True):
    # 4 labels, classes
    model = UNet(1, 4)
    
    if len(load) > 0:
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f'{load}.pth')))
    
    pipeline = Pipeline(model, dataset=dataset, visualize=visualize, outputs=outputs, load=load, total_epochs=total_epochs, start_epochs=start_epochs, batch_size=batch_size, has_labels=has_labels)
    
    if train==True:
        pipeline.train()
    pipeline.test()

if __name__ == '__main__':
    dataset = "../Training"
    outputs = 'segmentation'
    main(dataset, outputs, total_epochs=100, visualize="segmentationVisualization", start_epochs=0, batch_size=8)

    # 4 labels, classes
    #main(dataset, outputs, load=False, total_epochs=50, start_epochs=0, batch_size=8)