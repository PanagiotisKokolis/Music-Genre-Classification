import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics

def load_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    
    print("Data succesfully loaded!")
    return X, y


class MusicModel(pl.LightningModule):
    """Use high-level API known as Pytorch-Lightning to abstract the training and validation steps.
    """
    def __init__(self, input_shape):
        super(MusicModel, self).__init__()
        self.save_hyperparameters()
        self.flatten = nn.Flatten() # Flatten the input into 1D values
        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)
        
        # Dropout value of 0.3 means there's a 30% chance any particular neuron can be turned off during training.
        self.dropout = nn.Dropout(0.3)
        
        # A common metric to measure loss. Lower values means less error.
        # H(P, Q) = — sum x in X P(x) * log(Q(x))
        # Negative Log-Likelihood is another common loss function used.
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)


    def forward(self, x):
        """Use ReLU for deep learning, which is preferred over the classic Sigmoid function. You can read about it here:
        https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks
        The dropout function is applied to only the first two layers.
        """
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch # Split the current step's batch into the x and y values.
        y_hat = self(x) # The model's predicited y value.
        
        loss = self.criterion(y_hat, y)
        self.train_accuracy(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        self.val_accuracy(y_hat, y)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Use L2 regularization which attempts to reduce outlier weights
        return torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.001)

if __name__ == "__main__":
    X, y = load_data('data_10.json')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3) # Split the data into training data and validation data
    
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=32)

    input_shape = X.shape[1] * X.shape[2]
    lightning_model = MusicModel(input_shape)

    logger = TensorBoardLogger('tb_logs', name='Genre Classifier Model')
    trainer = pl.Trainer(max_epochs=100, logger=logger)
    trainer.fit(lightning_model, train_loader, val_loader)
    trainer.save_checkpoint("model_checkpoint.ckpt")
