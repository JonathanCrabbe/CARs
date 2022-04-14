import torch
import torch.nn as nn
import logging
import pathlib
import json
import numpy as np
from tqdm import tqdm
from utils.metrics import AverageMeter


class ClassifierMnist(nn.Module):
    def __init__(self, latent_dim: int, name: str = "model"):
        super(ClassifierMnist, self).__init__()
        self.latent_dim = latent_dim
        self.name = name
        self.checkpoints_files = []
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(32 * 4 * 4, 2*self.latent_dim)
        self.fc2 = nn.Linear(2*self.latent_dim, self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

    def input_to_representation(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def representation_to_output(self, h):
        h = self.dropout(h)
        h = self.out(h)
        return h

    def train_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer) -> np.ndarray:
        """
        One epoch of the training loop
        Args:
            device: device where tensor manipulations are done
            dataloader: training set dataloader
            optimizer: training optimizer

        Returns:
            average loss on the training set
        """
        self.train()
        train_loss = []
        loss_meter = AverageMeter("Loss")
        train_bar = tqdm(dataloader, unit="batch", leave=False)
        for image_batch, label_batch in train_bar:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            pred_batch = self.forward(image_batch)
            loss = self.criterion(pred_batch, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), len(image_batch))
            train_bar.set_description(f"Training Loss {loss_meter.avg:.3g}")
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader) -> tuple:
        """
        One epoch of the testing loop
        Args:
            device: device where tensor manipulations are done
            dataloader: test set dataloader

        Returns:
            average loss and accuracy on the training set
        """
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
                test_acc.append(
                    torch.count_nonzero(label_batch == torch.argmax(pred_batch, dim=-1)).cpu().numpy()/len(label_batch)
                                )

        return np.mean(test_loss), np.mean(test_acc)

    def fit(self, device: torch.device, train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader, save_dir: pathlib.Path,
            lr: int = 1e-03, n_epoch: int = 50, patience: int = 10, checkpoint_interval: int = -1) -> None:
        """
        Fit the classifier on the training set
        Args:
            device: device where tensor manipulations are done
            train_loader: training set dataloader
            test_loader: test set dataloader
            save_dir: path where checkpoints and model should be saved
            lr: learning rate
            n_epoch: maximum number of epochs
            patience: optimizer patience
            checkpoint_interval: number of epochs between each save

        Returns:

        """
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-05)
        waiting_epoch = 0
        best_test_loss = float("inf")
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss, test_acc = self.test_epoch(device, test_loader)
            logging.info(f'Epoch {epoch + 1}/{n_epoch} \t '
                         f'Train Loss {train_loss:.3g} \t '
                         f'Test Loss {test_loss:.3g} \t'
                         f'Test Accuracy {test_acc * 100:.3g}% \t ')
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

    def get_hooked_modules(self) -> dict[str, nn.Module]:
        return {"Conv1": self.maxpool1, "Conv2": self.maxpool2, "Lin1": self.fc1, "Lin2": self.fc2}