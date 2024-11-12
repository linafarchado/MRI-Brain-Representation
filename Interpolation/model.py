import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from trainAndTest import train, evaluate, test, test_random_images
from ConvAutoencoder import ConvAutoencoder

import sys
import torch.nn.functional as F

# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Interpolation import ImageInterpolator
from Dataloader import CustomDataset
from Utils import EarlyStopping

class Pipeline:
    def __init__(self, model, train_images, test_images, outputs, load="", batch_size=32, total_epochs=50, start_epochs=0):
        self.outputs = outputs
        os.makedirs(self.outputs, exist_ok=True)
        self.load = load if len(load) > 0 else outputs
        
        if train_images is not None:
            self.dataset = CustomDataset(train_images)

            self.train_dataset = CustomDataset(train_images, start_idx=0, end_idx=8)
            self.val_dataset = CustomDataset(train_images, start_idx=8, end_idx=11)
    
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        if test_images is not None:
            self.test_dataset = CustomDataset(test_images)
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.interpolator = ImageInterpolator(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.writer = SummaryWriter(f'runs/{outputs}')
        self.best_loss = float('inf')
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
            train_loss = train(self.model, self.train_dataset, self.optimizer, self.device)
            val_loss = evaluate(self.model, self.val_dataset, self.device)
            
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
        test(self.load, self.test_dataset, self.model, self.device)
        #test_random_images(self.load, self.test_dataset, self.model, self.device)

def main(train_images, test_images=None, outputs='interpolation', load="", total_epochs=50, batch_size=32):
    model = ConvAutoencoder(out_channels=32)
    
    if len(load) > 0:
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f'{load}.pth')))
    
    pipeline = Pipeline(
        model=model,
        train_images=train_images,
        test_images=test_images,
        outputs=outputs,
        load=load,
        total_epochs=total_epochs,
        batch_size=batch_size,
    )
    
    if train_images is not None:
        pipeline.trainAndEval()
    
    if test_images is not None:
        pipeline.test()

if __name__ == '__main__':
    train_images = "../Training"
    test_images = "../Testing"
    outputs = "interpolation"
    main(train_images=None, test_images=test_images, outputs=outputs, load=outputs, total_epochs=100, batch_size=16)