import torch
import torch.nn as nn
import logging
import pathlib
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class ClassifierMnist(nn.Module):
    def __init__(self, latent_dim: int, name: str = "model"):
        super(ClassifierMnist, self).__init__()
        self.latent_dim = latent_dim
        self.name = name
        self.checkpoints_files = []
        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.cnn_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(32 * 4 * 4, 2*self.latent_dim)
        self.fc2 = nn.Linear(2*self.latent_dim, self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.cnn_1(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.maxpool(x)
        x = self.cnn_2(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

    def train_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer) -> np.ndarray:
        self.train()
        train_loss = []
        for image_batch, label_batch in tqdm(dataloader, unit="batch", leave=False):
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            pred_batch = self.forward(image_batch)
            loss = self.criterion(pred_batch, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader) -> tuple:
        self.eval()
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for image_batch, label_batch in dataloader:
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)
                pred_batch = self.forward(image_batch)
                loss = self.criterion(pred_batch, label_batch)
                test_loss.append(loss.cpu().numpy())
                test_acc.append(accuracy_score(label_batch.cpu().numpy(), pred_batch.cpu().numpy()))
        return np.mean(test_loss), np.mean(test_acc)

    def fit(self, device: torch.device, train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader, save_dir: pathlib.Path,
            lr: int = 1e-03, n_epoch: int = 30, patience: int = 10, checkpoint_interval: int = -1) -> None:
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-05)
        waiting_epoch = 0
        best_test_loss = float("inf")
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss = self.test_epoch(device, test_loader)
            logging.info(f'Epoch {epoch + 1}/{n_epoch} \t '
                         f'Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t ')
            if test_loss >= best_test_loss:
                waiting_epoch += 1
                logging.info(f'No improvement over the best epoch \t Patience {waiting_epoch} / {patience}')
            else:
                logging.info(f'Saving the model in {save_dir}')
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_loss = test_loss.data
                waiting_epoch = 0
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                logging.info(f'Saving checkpoint {n_checkpoint} in {save_dir}')
                path_to_checkpoint = save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
                torch.save(self.state_dict(), path_to_checkpoint)
                self.checkpoints_files.append(path_to_checkpoint)
            if waiting_epoch == patience:
                logging.info(f'Early stopping activated')
                break

    def save(self, directory: pathlib.Path) -> None:
        """
        Save a model and corresponding metadata.
        Parameters
        ----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        self.save_metadata(directory)
        path_to_model = directory / (self.name + ".pt")
        torch.save(self.state_dict(), path_to_model)

    def load_metadata(self, directory: pathlib.Path) -> dict:
        """Load the metadata of a training directory.
        Parameters
        ----------
        directory : pathlib.Path
            Path to folder where model is saved. For example './experiments/mnist'.
        """
        path_to_metadata = directory / (self.name + ".json")

        with open(path_to_metadata) as metadata_file:
            metadata = json.load(metadata_file)
        return metadata

    def save_metadata(self, directory: pathlib.Path, **kwargs) -> None:
        """Load the metadata of a training directory.
        Parameters
        ----------
        directory: string
            Path to folder where to save model. For example './experiments/mnist'.
        kwargs:
            Additional arguments to `json.dump`
        """
        path_to_metadata = directory / (self.name + ".json")
        metadata = {"latent_dim": self.latent_dim,
                    "name": self.name,
                    "checkpoint_files": self.checkpoints_files}
        with open(path_to_metadata, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)