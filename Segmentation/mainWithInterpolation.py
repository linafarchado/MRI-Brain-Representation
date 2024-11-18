import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from trainAndTest import train, evaluate, test
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import sys
import os
from model import UNet

# Add parent directory to sys.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Dataloader import CustomDatasetWithLabelsFiltered, CustomDataset
from Utils import EarlyStopping, save_label_nifti, save_image_nifti
from Metrics import calculate_dice_per_class
from Visualize import visualize_segmentation_Dice, plot_dice_vs_std

class Pipeline():
    def __init__(self, model, visualize, outputs, load, original_images, artificialFIRST, artificialSEC, test_images, batch_size=8, start_epochs=0, total_epochs=50, artificial_weight=0.5, has_labels=True):
        self.outputs = outputs
        self.visualize = visualize
        self.has_labels = has_labels
        os.makedirs(self.outputs, exist_ok=True)
        os.makedirs(self.visualize, exist_ok=True)
        self.load = load if len(load) > 0 else outputs
        self.artificial_weight = artificial_weight

        # Load original and artificial datasets
        self.original_dataset = CustomDatasetWithLabelsFiltered(original_images, is_training=True)
        self.artificial_dataset_first = CustomDatasetWithLabelsFiltered(artificialFIRST, interpolation=True)
        if artificialSEC is not None:
            self.artificial_dataset_sec = CustomDatasetWithLabelsFiltered(artificialSEC, interpolation=True)
            self.artificial_dataset = ConcatDataset([self.artificial_dataset_first, self.artificial_dataset_sec])
        else:
            self.artificial_dataset = self.artificial_dataset_first
        # Combine datasets
        combined_dataset = ConcatDataset([self.original_dataset, self.artificial_dataset])
        
        # Split combined dataset into train and validation
        train_size = int(0.8 * len(combined_dataset))
        val_size = len(combined_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])

        if test_images is not None:
            self.test_dataset = CustomDatasetWithLabelsFiltered(test_images, is_training=False) if self.has_labels else CustomDataset(test_images)
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size)
        
        print(f'Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}')
        print(f'Artificial size: {len(self.artificial_dataset)}')
        print(f'Original size: {len(self.original_dataset)}')

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

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

    def save_best_model(self, epoch, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model = self.model.state_dict()
            model_path = os.path.join(self.outputs, f'best_model_epoch_{epoch}.pth')
            torch.save(self.best_model, model_path)

    

    def trainAndEval(self):
        for epoch in range(self.start_epochs, self.total_epochs):
            self.model.train()
            train_loss = 0
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.total_epochs}"):
                # Clip labels to 0-3
                labels = torch.clamp(labels, 0, 3)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                # Apply different weights to original and artificial data
                if images in self.original_dataset:
                    loss = self.criterion(outputs, labels)
                else:
                    loss = self.artificial_weight * self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            val_loss = evaluate(self.model, self.val_loader, self.criterion, self.device)
            
            print(f'Epoch {epoch+1}/{self.total_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"Early stopping, epoch: {epoch}")
                break

            self.save_best_model(epoch, val_loss)

        self.writer.close()
        torch.save(self.model.state_dict(), f'{self.outputs}.pth')

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f'{self.load}.pth')))
        if self.has_labels:
            preds, labels = test(self.model, self.test_loader, self.device)

            accuracy = np.mean(preds == labels)
            print(f'Accuracy: {accuracy:.4f}')
            # save accuracy in a file
            with open(os.path.join(self.visualize, 'accuracy.txt'), 'w') as f:
                f.write(f'Accuracy: {accuracy:.4f}')

            dice_scores = []
            std_values = []

            for i in range(len(self.test_dataset)):
                test_image, test_label = self.test_dataset[i]
                test_image = test_image.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    test_output = self.model(test_image)
                    test_pred = torch.argmax(test_output, dim=1).squeeze().cpu().numpy()
                    dice_score = calculate_dice_per_class(torch.from_numpy(test_pred).long(), test_label, 4)
                    dice_mean = np.mean([score for cls, score in dice_score])
                    dice_scores.append(dice_mean)
                
                    std_value = np.std(test_pred)
                    std_values.append(std_value)
            
            # Tracer le graphe
            plot_dice_vs_std(dice_scores, std_values, save_path=os.path.join(self.visualize, 'dice_vs_std.png'))


            # Visualize 50 random samples
            
            for i in range(len(self.test_dataset)):
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
            #self.visualize_predictions()
            self.save_predictions()

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

    def save_predictions(self):
        self.model.eval()
        os.makedirs(self.visualize, exist_ok=True)

        for i in range(len(self.test_dataset)):
            test_image = self.test_dataset[i]
            test_image = test_image.unsqueeze(0).to(self.device)

            test_output = self.model(test_image)
            test_pred = torch.argmax(test_output, dim=1).squeeze().cpu().numpy()
            """
            pred_3d = np.clip(test_pred * 255, 0, 255).astype(np.uint8)
            pred_3d = np.expand_dims(pred_3d, axis=2)
            save_image_nifti(test_image_np, f'interpolated-{i}',self.visualize)
            save_label_nifti(pred_3d, f'interpolated-{i}',self.visualize)"""
            test_image_np = test_image.squeeze().cpu().numpy()

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(test_image_np, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(test_pred, cmap='viridis')
            plt.title('Predicted Segmentation')
            plt.axis('off')
            
            plt.savefig(os.path.join(self.visualize, f'prediction_{i+1}.png'))
            plt.close()
            
    def visualize_predictions(self, num_samples=100):
        os.makedirs(self.visualize, exist_ok=True)
        images, predictions = self.inference()
        
        #for i in range(min(num_samples, len(images))):
        for i in range(len(images)):
            image = images[i]
            pred = predictions[i]
            """
            # To save the predicted labels as nifti files

            pred_3d = np.clip(pred * 255, 0, 255).astype(np.uint8)
            pred_3d = np.expand_dims(pred_3d, axis=2)
            save_image_nifti(image.squeeze(), f'interpolated-{i}',self.visualize)
            save_label_nifti(pred_3d, f'interpolated-{i}',self.visualize)"""

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(pred, cmap='viridis')
            plt.title('Predicted Segmentation')
            plt.axis('off')
            
            plt.savefig(os.path.join(self.visualize, f'prediction_{i+1}.png'))
            plt.close()


def main():
    model = UNet(1, 4)
    pipeline = Pipeline(model, visualize='NewSegmenter', outputs='newsegmenter', load='newsegmenter', original_images='../Training', artificial_images='../InterpolationSavedLabels', test_images='../Training', batch_size=8, start_epochs=0, total_epochs=100, artificial_weight=0.5, has_labels=True)
    #pipeline.trainAndEval()
    pipeline.test()

def main_different_weigth(training, artificialFIRST, artificialSEC, testing, load=''):
    weights = [0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9, 0.95]
    for weight in weights:
        model = UNet(1, 4)
        pipeline = Pipeline(
            model,
            visualize=f'NewSegmenter_{weight}',
            outputs=f'newsegmenter_{weight}',
            load=load,
            original_images=training,
            artificialFIRST=artificialFIRST,
            artificialSEC=artificialSEC,
            test_images=testing,
            batch_size=8,
            start_epochs=0,
            total_epochs=100,
            artificial_weight=weight,
            has_labels=True
        )
        pipeline.trainAndEval()
        pipeline.test()

if __name__ == '__main__':
    training = '../Training'
    artificialFIRST = '../InterpolationSavedLabelsMulti'
    artificialSEC = None
    testing = '../Testing'
    main_different_weigth(training, artificialFIRST, artificialSEC, training)