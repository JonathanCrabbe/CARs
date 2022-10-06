import torch
import torch.nn as nn
import logging
import pathlib
import json
import os
import numpy as np
import csv
import torch.nn.functional as F
import torch.optim as opt
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
from utils.metrics import AverageMeter
from abc import abstractmethod
from os import path
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

BEST_MODEL_FILENAME = "best_model.pt"


class ClassifierMnist(nn.Module):
    def __init__(self, latent_dim: int, name: str = "model"):
        super(ClassifierMnist, self).__init__()
        self.latent_dim = latent_dim
        self.name = name
        self.checkpoints_files = []
        self.cnn1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        self.cnn2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0
        )
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(32 * 4 * 4, 2 * self.latent_dim)
        self.fc2 = nn.Linear(2 * self.latent_dim, self.latent_dim)
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

    def train_epoch(
        self,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> np.ndarray:
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

    def test_epoch(
        self, device: torch.device, dataloader: torch.utils.data.DataLoader
    ) -> tuple:
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
                    torch.count_nonzero(label_batch == torch.argmax(pred_batch, dim=-1))
                    .cpu()
                    .numpy()
                    / len(label_batch)
                )

        return np.mean(test_loss), np.mean(test_acc)

    def fit(
        self,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        save_dir: pathlib.Path,
        lr: int = 1e-03,
        n_epoch: int = 50,
        patience: int = 10,
        checkpoint_interval: int = -1,
    ) -> None:
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
            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train Loss {train_loss:.3g} \t "
                f"Test Loss {test_loss:.3g} \t"
                f"Test Accuracy {test_acc * 100:.3g}% \t "
            )
            if test_loss >= best_test_loss:
                waiting_epoch += 1
                logging.info(
                    f"No improvement over the best epoch \t Patience {waiting_epoch} / {patience}"
                )
            else:
                logging.info(f"Saving the model in {save_dir}")
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_loss = test_loss.data
                waiting_epoch = 0
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                logging.info(f"Saving checkpoint {n_checkpoint} in {save_dir}")
                path_to_checkpoint = (
                    save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
                )
                torch.save(self.state_dict(), path_to_checkpoint)
                self.checkpoints_files.append(path_to_checkpoint)
            if waiting_epoch == patience:
                logging.info(f"Early stopping activated")
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
        metadata = {
            "latent_dim": self.latent_dim,
            "name": self.name,
            "checkpoint_files": self.checkpoints_files,
        }
        with open(path_to_metadata, "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)

    def get_hooked_modules(self) -> dict[str, nn.Module]:
        return {
            "Conv1": self.maxpool1,
            "Conv2": self.maxpool2,
            "Lin1": self.fc1,
            "Lin2": self.fc2,
        }


class SENN(nn.Module):
    def __init__(self, conceptizer, parameterizer, aggregator):
        """Represents a Self Explaining Neural Network (SENN).
        Code adapted from https://github.com/AmanDaVinci/SENN
        (https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks)
        A SENN model is a neural network made explainable by design. It is made out of several submodules:
            - conceptizer
                Model that encodes raw input into interpretable feature representations of
                that input. These feature representations are called concepts.
            - parameterizer
                Model that computes the parameters theta from given the input. Each concept
                has with it associated one theta, which acts as a ``relevance score'' for that concept.
            - aggregator
                Predictions are made with a function g(theta_1 * h_1, ..., theta_n * h_n), where
                h_i represents concept i. The aggregator defines the function g, i.e. how each
                concept with its relevance score is combined into a prediction.
        Parameters
        ----------
        conceptizer : Pytorch Module
            Model that encodes raw input into interpretable feature representations of
            that input. These feature representations are called concepts.
        parameterizer : Pytorch Module
            Model that computes the parameters theta from given the input. Each concept
            has with it associated one theta, which acts as a ``relevance score'' for that concept.
        aggregator : Pytorch Module
            Predictions are made with a function g(theta_1 * h_1, ..., theta_n * h_n), where
            h_i represents concept i. The aggregator defines the function g, i.e. how each
            concept with its relevance score is combined into a prediction.
        """
        super().__init__()
        self.conceptizer = conceptizer
        self.parameterizer = parameterizer
        self.aggregator = aggregator

    def forward(self, x):
        """Forward pass of SENN module.

        In the forward pass, concepts and their reconstructions are created from the input x.
        The relevance parameters theta are also computed.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.
        Returns
        -------
        predictions : torch.Tensor
            Predictions generated by model. Of shape (BATCH, *).

        explanations : tuple
            Model explanations given by a tuple (concepts, relevances).
            concepts : torch.Tensor
                Interpretable feature representations of input. Of shape (NUM_CONCEPTS, *).
            parameters : torch.Tensor
                Relevance scores associated with concepts. Of shape (NUM_CONCEPTS, *)
        """
        concepts, recon_x = self.conceptizer(x)
        relevances = self.parameterizer(x)
        predictions = self.aggregator(concepts, relevances)
        explanations = (concepts, relevances)
        return predictions, explanations, recon_x


class SumAggregator(nn.Module):
    def __init__(self, num_classes, **kwargs):
        """Basic Sum Aggregator that joins the concepts and relevances by summing their products."""
        super().__init__()
        self.num_classes = num_classes

    def forward(self, concepts, relevances):
        """Forward pass of Sum Aggregator.
        Aggregates concepts and relevances and returns the predictions for each class.
        Parameters
        ----------
        concepts : torch.Tensor
            Contains the output of the conceptizer with shape (BATCH, NUM_CONCEPTS, DIM_CONCEPT=1).
        relevances : torch.Tensor
            Contains the output of the parameterizer with shape (BATCH, NUM_CONCEPTS, NUM_CLASSES).
        Returns
        -------
        class_predictions : torch.Tensor
            Predictions for each class. Shape - (BATCH, NUM_CLASSES)

        """
        aggregated = torch.bmm(relevances.permute(0, 2, 1), concepts).squeeze(-1)
        return F.log_softmax(aggregated, dim=1)


class Conceptizer(nn.Module):
    def __init__(self):
        """
        A general Conceptizer meta-class. Children of the Conceptizer class
        should implement encode() and decode() functions.
        """
        super(Conceptizer, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

    def forward(self, x):
        """
        Forward pass of the general conceptizer.
        Computes concepts present in the input.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.
        Returns
        -------
        encoded : torch.Tensor
            Encoded concepts (batch_size, concept_number, concept_dimension)
        decoded : torch.Tensor
            Reconstructed input (batch_size, *)
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded.view_as(x)

    @abstractmethod
    def encode(self, x):
        """
        Abstract encode function to be overridden.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.
        """
        pass

    @abstractmethod
    def decode(self, encoded):
        """
        Abstract decode function to be overridden.
        Parameters
        ----------
        encoded : torch.Tensor
            Latent representation of the data
        """
        pass


class ConvConceptizer(Conceptizer):
    def __init__(
        self,
        image_size,
        num_concepts,
        concept_dim,
        image_channels=1,
        encoder_channels=(10,),
        decoder_channels=(16, 8),
        kernel_size_conv=5,
        kernel_size_upsample=(5, 5, 2),
        stride_conv=1,
        stride_pool=2,
        stride_upsample=(2, 1, 2),
        padding_conv=0,
        padding_upsample=(0, 0, 1),
        **kwargs,
    ):
        """
        CNN Autoencoder used to learn the concepts, present in an input image
        Parameters
        ----------
        image_size : int
            the width of the input image
        num_concepts : int
            the number of concepts
        concept_dim : int
            the dimension of each concept to be learned
        image_channels : int
            the number of channels of the input images
        encoder_channels : tuple[int]
            a list with the number of channels for the hidden convolutional layers
        decoder_channels : tuple[int]
            a list with the number of channels for the hidden upsampling layers
        kernel_size_conv : int, tuple[int]
            the size of the kernels to be used for convolution
        kernel_size_upsample : int, tuple[int]
            the size of the kernels to be used for upsampling
        stride_conv : int, tuple[int]
            the stride of the convolutional layers
        stride_pool : int, tuple[int]
            the stride of the pooling layers
        stride_upsample : int, tuple[int]
            the stride of the upsampling layers
        padding_conv : int, tuple[int]
            the padding to be used by the convolutional layers
        padding_upsample : int, tuple[int]
            the padding to be used by the upsampling layers
        """
        super(ConvConceptizer, self).__init__()
        self.num_concepts = num_concepts
        self.filter = filter
        self.dout = image_size

        # Encoder params
        encoder_channels = (image_channels,) + encoder_channels
        kernel_size_conv = handle_integer_input(kernel_size_conv, len(encoder_channels))
        stride_conv = handle_integer_input(stride_conv, len(encoder_channels))
        stride_pool = handle_integer_input(stride_pool, len(encoder_channels))
        padding_conv = handle_integer_input(padding_conv, len(encoder_channels))
        encoder_channels += (num_concepts,)

        # Decoder params
        decoder_channels = (num_concepts,) + decoder_channels
        kernel_size_upsample = handle_integer_input(
            kernel_size_upsample, len(decoder_channels)
        )
        stride_upsample = handle_integer_input(stride_upsample, len(decoder_channels))
        padding_upsample = handle_integer_input(padding_upsample, len(decoder_channels))
        decoder_channels += (image_channels,)

        # Encoder implementation
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoder.append(
                self.conv_block(
                    in_channels=encoder_channels[i],
                    out_channels=encoder_channels[i + 1],
                    kernel_size=kernel_size_conv[i],
                    stride_conv=stride_conv[i],
                    stride_pool=stride_pool[i],
                    padding=padding_conv[i],
                )
            )
            self.dout = (
                self.dout
                - kernel_size_conv[i]
                + 2 * padding_conv[i]
                + stride_conv[i] * stride_pool[i]
            ) // (stride_conv[i] * stride_pool[i])

        if self.filter and concept_dim == 1:
            self.encoder.append(
                ScalarMapping((self.num_concepts, self.dout, self.dout))
            )
        else:
            self.encoder.append(Flatten())
            self.encoder.append(nn.Linear(self.dout**2, concept_dim))

        # Decoder implementation
        self.unlinear = nn.Linear(concept_dim, self.dout**2)
        self.decoder = nn.ModuleList()
        decoder = []
        for i in range(len(decoder_channels) - 1):
            decoder.append(
                self.upsample_block(
                    in_channels=decoder_channels[i],
                    out_channels=decoder_channels[i + 1],
                    kernel_size=kernel_size_upsample[i],
                    stride_deconv=stride_upsample[i],
                    padding=padding_upsample[i],
                )
            )
            decoder.append(nn.ReLU(inplace=True))
        decoder.pop()
        decoder.append(nn.Tanh())
        self.decoder = nn.ModuleList(decoder)

    def encode(self, x):
        """
        The encoder part of the autoencoder which takes an Image as an input
        and learns its hidden representations (concepts)
        Parameters
        ----------
        x : Image (batch_size, channels, width, height)
        Returns
        -------
        encoded : torch.Tensor (batch_size, concept_number, concept_dimension)
            the concepts representing an image
        """
        encoded = x
        for module in self.encoder:
            encoded = module(encoded)
        return encoded

    def decode(self, z):
        """
        The decoder part of the autoencoder which takes a hidden representation as an input
        and tries to reconstruct the original image
        Parameters
        ----------
        z : torch.Tensor (batch_size, channels, width, height)
            the concepts in an image
        Returns
        -------
        reconst : torch.Tensor (batch_size, channels, width, height)
            the reconstructed image
        """
        reconst = self.unlinear(z)
        reconst = reconst.view(-1, self.num_concepts, self.dout, self.dout)
        for module in self.decoder:
            reconst = module(reconst)
        return reconst

    def conv_block(
        self, in_channels, out_channels, kernel_size, stride_conv, stride_pool, padding
    ):
        """
        A helper function that constructs a convolution block with pooling and activation
        Parameters
        ----------
        in_channels : int
            the number of input channels
        out_channels : int
            the number of output channels
        kernel_size : int
            the size of the convolutional kernel
        stride_conv : int
            the stride of the deconvolution
        stride_pool : int
            the stride of the pooling layer
        padding : int
            the size of padding
        Returns
        -------
        sequence : nn.Sequence
            a sequence of convolutional, pooling and activation modules
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride_conv,
                padding=padding,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=stride_pool, padding=padding),
            nn.ReLU(inplace=True),
        )

    def upsample_block(
        self, in_channels, out_channels, kernel_size, stride_deconv, padding
    ):
        """
        A helper function that constructs an upsampling block with activations
        Parameters
        ----------
        in_channels : int
            the number of input channels
        out_channels : int
            the number of output channels
        kernel_size : int
            the size of the convolutional kernel
        stride_deconv : int
            the stride of the deconvolution
        padding : int
            the size of padding
        Returns
        -------
        sequence : nn.Sequence
            a sequence of deconvolutional and activation modules
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride_deconv,
                padding=padding,
            ),
        )


class Flatten(nn.Module):
    def forward(self, x):
        """
        Flattens the inputs to only 3 dimensions, preserving the sizes of the 1st and 2nd.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (dim1, dim2, *).
        Returns
        -------
        flattened : torch.Tensor
            Flattened input (dim1, dim2, dim3)
        """
        return x.view(x.size(0), x.size(1), -1)


class ScalarMapping(nn.Module):
    def __init__(self, conv_block_size):
        """
        Module that maps each filter of a convolutional block to a scalar value
        Parameters
        ----------
        conv_block_size : tuple (int iterable)
            Specifies the size of the input convolutional block: (NUM_CHANNELS, FILTER_HEIGHT, FILTER_WIDTH)
        """
        super().__init__()
        self.num_filters, self.filter_height, self.filter_width = conv_block_size

        self.layers = nn.ModuleList()
        for _ in range(self.num_filters):
            self.layers.append(nn.Linear(self.filter_height * self.filter_width, 1))

    def forward(self, x):
        """
        Reduces a 3D convolutional block to a 1D vector by mapping each 2D filter to a scalar value.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, CHANNELS, HEIGHT, WIDTH).
        Returns
        -------
        mapped : torch.Tensor
            Reduced input (BATCH, CHANNELS, 1)
        """
        x = x.view(-1, self.num_filters, self.filter_height * self.filter_width)
        mappings = []
        for f, layer in enumerate(self.layers):
            mappings.append(layer(x[:, [f], :]))
        return torch.cat(mappings, dim=1)


class ConvParameterizer(nn.Module):
    def __init__(
        self,
        num_concepts,
        num_classes,
        cl_sizes=(1, 10, 20),
        kernel_size=5,
        hidden_sizes=(10, 5, 5, 10),
        dropout=0.5,
        **kwargs,
    ):
        """Parameterizer for MNIST dataset.
        Consists of convolutional as well as fully connected modules.
        Parameters
        ----------
        num_concepts : int
            Number of concepts that should be parameterized (for which the relevances should be determined).
        num_classes : int
            Number of classes that should be distinguished by the classifier.
        cl_sizes : iterable of int
            Indicates the number of kernels of each convolutional layer in the network. The first element corresponds to
            the number of input channels.
        kernel_size : int
            Indicates the size of the kernel window for the convolutional layers.
        hidden_sizes : iterable of int
            Indicates the size of each fully connected layer in the network. The first element corresponds to
            the number of input features. The last element must be equal to the number of concepts multiplied with the
            number of output classes.
        dropout : float
            Indicates the dropout probability.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.cl_sizes = cl_sizes
        self.kernel_size = kernel_size
        self.dropout = dropout

        cl_layers = []
        for h, h_next in zip(cl_sizes, cl_sizes[1:]):
            cl_layers.append(nn.Conv2d(h, h_next, kernel_size=self.kernel_size))
            # TODO: maybe adaptable parameters for pool kernel size and stride
            cl_layers.append(nn.MaxPool2d(2, stride=2))
            cl_layers.append(nn.ReLU())
        # dropout before maxpool
        cl_layers.insert(-2, nn.Dropout2d(self.dropout))
        self.cl_layers = nn.Sequential(*cl_layers)

        fc_layers = []
        for h, h_next in zip(hidden_sizes, hidden_sizes[1:]):
            fc_layers.append(nn.Linear(h, h_next))
            fc_layers.append(nn.Dropout(self.dropout))
            fc_layers.append(nn.ReLU())
        fc_layers.pop()
        fc_layers.append(nn.Tanh())
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        """Forward pass of MNIST parameterizer.
        Computes relevance parameters theta.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.
        Returns
        -------
        parameters : torch.Tensor
            Relevance scores associated with concepts. Of shape (BATCH, NUM_CONCEPTS, NUM_CLASSES)
        """
        cl_output = self.cl_layers(x)
        flattened = cl_output.view(x.size(0), -1)
        return self.fc_layers(flattened).view(-1, self.num_concepts, self.num_classes)


class SENN_Trainer:
    def __init__(self, config):
        """Base SENN Trainer class.

        A trainer instantiates a model to be trained. It contains logic for training, validating,
        checkpointing, etc. All the specific parameters that control the experiment behaviour
        are contained in the configs json.
        The models we consider here are all Self Explaining Neural Networks (SENNs).
        If `load_checkpoint` is specified in configs and the model has a checkpoint, the checkpoint
        will be loaded.

        Parameters
        ----------
        config : types.SimpleNamespace
            Contains all (hyper)parameters that define the behavior of the program.
        """
        self.config = config
        logging.info(f"Using device {config.device}")

        # Load data
        logging.info("Loading data ...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(config)

        if hasattr(config, "manual_seed"):
            torch.manual_seed(config.manual_seed)

        # get appropriate models from global namespace and instantiate them
        try:
            conceptizer = eval(config.conceptizer)(**config.__dict__)
            parameterizer = eval(config.parameterizer)(**config.__dict__)
            aggregator = eval(config.aggregator)(**config.__dict__)
        except:
            logging.info(
                "Please make sure you specify the correct Conceptizer, Parameterizer and Aggregator classes"
            )
            exit(-1)

        # Define losses
        self.classification_loss = F.nll_loss
        self.concept_loss = mse_l1_sparsity
        self.robustness_loss = eval(config.robustness_loss)

        # Init model
        self.model = SENN(conceptizer, parameterizer, aggregator)
        self.model.to(config.device)

        # Init optimizer
        self.opt = opt.Adam(self.model.parameters(), lr=config.lr)

        # Init trackers
        self.current_iter = 0
        self.current_epoch = 0
        self.best_accuracy = 0

        # directories for saving results
        RESULTS_DIR = pathlib.Path.cwd() / "results/mnist/senn"
        CHECKPOINT_DIR = pathlib.Path.cwd() / "results/mnist/senn/checkpoints"
        LOG_DIR = pathlib.Path.cwd() / "results/mnist/senn/log"

        self.experiment_dir = path.join(RESULTS_DIR, config.exp_name)
        self.checkpoint_dir = path.join(self.experiment_dir, CHECKPOINT_DIR)
        self.log_dir = path.join(self.experiment_dir, LOG_DIR)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        if hasattr(config, "load_checkpoint"):
            self.load_checkpoint(config.load_checkpoint)

    def run(self):
        """Run the training loop.

        If the loop is interrupted manually, finalization will still be executed.
        """
        try:
            if self.config.train:
                logging.info("Training begins...")
                self.train()
        except KeyboardInterrupt:
            logging.info("CTRL+C pressed... Waiting to finalize.")

    def train(self):
        """Main training loop. Saves a model checkpoint after every epoch."""
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            self.train_one_epoch(self.current_epoch)
            self.save_checkpoint()

    def train_one_epoch(self, epoch):
        """Run one epoch of training.
        Parameters
        ----------
        epoch : int
            Current epoch.
        """
        self.model.train()

        for i, (x, labels) in enumerate(self.train_loader):
            x = x.float().to(self.config.device)
            labels = labels.long().to(self.config.device)
            self.opt.zero_grad()
            # track all operations on x for jacobian calculation
            x.requires_grad_(True)

            # run x through SENN
            y_pred, (concepts, relevances), x_reconstructed = self.model(x)

            classification_loss = self.classification_loss(y_pred.squeeze(-1), labels)
            robustness_loss = self.robustness_loss(x, y_pred, concepts, relevances)
            concept_loss = self.concept_loss(
                x, x_reconstructed, concepts, self.config.sparsity_reg
            )

            total_loss = (
                classification_loss
                + self.config.robust_reg * robustness_loss
                + self.config.concept_reg * concept_loss
            )
            total_loss.backward()
            self.opt.step()

            accuracy = self.accuracy(y_pred, labels)

            # --- Report Training Progress --- #
            self.current_iter += 1

            if i % self.config.print_freq == 0:
                logging.info(f"EPOCH:{epoch} STEP:{i}")
                self.print_n_save_metrics(
                    filename="accuracies_losses_train.csv",
                    total_loss=total_loss.item(),
                    classification_loss=classification_loss.item(),
                    robustness_loss=robustness_loss.item(),
                    concept_loss=concept_loss.item(),
                    accuracy=accuracy,
                )

            if self.current_iter % self.config.eval_freq == 0:
                self.validate()

    def validate(self):
        """Get the metrics for the validation set"""
        return self.get_metrics(validate=True)

    def test(self):
        """Get the metrics for the test set"""
        return self.get_metrics(validate=False)

    def get_metrics(self, validate=True):
        """Get the metrics for a validation/test set
        If the validation flag is on, the function tests the model
        with the validation dataset instead of the testing one.
        Model performance is validated by computing loss and accuracy measures, storing them,
        and reporting them.
        Parameters
        ----------
        validate : bool
            Indicates whether to use the validation or test dataset
        """
        losses_val = []
        classification_losses_val = []
        concept_losses_val = []
        robustness_losses_val = []
        accuracies_val = []

        dl = self.val_loader if validate else self.test_loader

        self.model.eval()
        with torch.no_grad():
            for x, labels in dl:
                x = x.float().to(self.config.device)
                labels = labels.long().to(self.config.device)

                # run x through SENN
                y_pred, (concepts, _), x_reconstructed = self.model(x)

                classification_loss = self.classification_loss(
                    y_pred.squeeze(-1), labels
                )
                # robustness_loss = self.robustness_loss(x, y_pred, concepts, relevances)
                robustness_loss = torch.tensor(
                    0.0
                )  # jacobian cannot be computed with no_grad enabled
                concept_loss = self.concept_loss(
                    x, x_reconstructed, concepts, self.config.sparsity_reg
                )

                total_loss = (
                    classification_loss
                    + self.config.robust_reg * robustness_loss
                    + self.config.concept_reg * concept_loss
                )

                accuracy = self.accuracy(y_pred, labels)

                losses_val.append(total_loss.item())
                classification_losses_val.append(classification_loss.item())
                concept_losses_val.append(concept_loss.item())
                robustness_losses_val.append(robustness_loss.item())
                accuracies_val.append(accuracy)

            classification_loss = np.mean(classification_losses_val)
            robustness_loss = np.mean(robustness_losses_val)
            concept_loss = np.mean(concept_losses_val)
            total_loss = np.mean(losses_val)
            accuracy = np.mean(accuracies_val)

            # --- Report statistics --- #
            logging.info(
                f"\n\033[93m-------- {'Validation' if validate else 'Test'} --------"
            )
            self.print_n_save_metrics(
                filename=f"accuracies_losses_{'valid' if validate else 'test'}.csv",
                total_loss=total_loss,
                classification_loss=classification_loss,
                robustness_loss=robustness_loss,
                concept_loss=concept_loss,
                accuracy=accuracy,
            )
            logging.info("----------------------------\033[0m")

            if accuracy > self.best_accuracy and validate:
                logging.info(
                    "\033[92mCongratulations! Saving a new best model...\033[00m"
                )
                self.best_accuracy = accuracy
                self.save_checkpoint(BEST_MODEL_FILENAME)

        return accuracy

    def accuracy(self, y_pred, y):
        """Return accuracy of predictions with respect to ground truth.
        Parameters
        ----------
        y_pred : torch.Tensor, shape (BATCH,)
            Predictions of ground truth.
        y : torch.Tensor, shape (BATCH,)
            Ground truth.
        Returns
        -------
        float:
            accuracy of predictions
        """
        return (y_pred.argmax(axis=1) == y).float().mean().item()

    def load_checkpoint(self, file_name):
        """Load most recent checkpoint.
        If no checkpoint exists, doesn't do anything.
        Checkpoint contains:
            - current epoch
            - current iteration
            - model state
            - best accuracy achieved so far
            - optimizer state
        Parameters
        ----------
        file_name: str
            Name of the checkpoint file.
        """
        try:
            file_name = path.join(self.checkpoint_dir, file_name)
            logging.info(f"Loading checkpoint...")
            checkpoint = torch.load(file_name, self.config.device)

            self.current_epoch = checkpoint["epoch"]
            self.current_iter = checkpoint["iter"]
            self.best_accuracy = checkpoint["best_accuracy"]
            self.model.load_state_dict(checkpoint["model_state"])
            self.opt.load_state_dict(checkpoint["optimizer"])

            logging.info(f"Checkpoint loaded successfully from '{file_name}'\n")

        except OSError:
            logging.info(f"No checkpoint exists @ {self.checkpoint_dir}")
            logging.info("**Training for the first time**")

    def save_checkpoint(self, file_name=None):
        """Save checkpoint in the checkpoint directory.
        Checkpoint dir and checkpoint_file need to be specified in the configs.
        Parameters
        ----------
        file_name: str
            Name of the checkpoint file.
        """
        if file_name is None:
            file_name = f"Epoch[{self.current_epoch}]-Step[{self.current_iter}].pt"

        file_name = path.join(self.checkpoint_dir, file_name)
        state = {
            "epoch": self.current_epoch,
            "iter": self.current_iter,
            "best_accuracy": self.best_accuracy,
            "model_state": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
        }
        torch.save(state, file_name)

        logging.info(f"Checkpoint saved @ {file_name}\n")

    def print_n_save_metrics(
        self,
        filename,
        total_loss,
        classification_loss,
        robustness_loss,
        concept_loss,
        accuracy,
    ):
        """Prints the losses to the console and saves them in a csv file
        Parameters
        ----------
        filename: str
            Name of the csv file.
        classification_loss: float
            The value of the classification loss
        robustness_loss: float
            The value of the robustness loss
        total_loss: float
            The value of the total loss
        concept_loss: float
            The value of the concept loss
        accuracy: float
            The value of the accuracy
        """
        report = (
            f"Total Loss:{total_loss:.3f} \t"
            f"Classification Loss:{classification_loss:.3f} \t"
            f"Robustness Loss:{robustness_loss:.3f} \t"
            f"Concept Loss:{concept_loss:.3f} \t"
            f"Accuracy:{accuracy:.3f} \t"
        )
        logging.info(report)

        filename = path.join(self.experiment_dir, filename)
        new_file = not os.path.exists(filename)
        with open(filename, "a") as metrics_file:
            fieldnames = [
                "Accuracy",
                "Loss",
                "Classification_Loss",
                "Robustness_Loss",
                "Concept_Loss",
                "Step",
            ]
            csv_writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)

            if new_file:
                csv_writer.writeheader()

            csv_writer.writerow(
                {
                    "Accuracy": accuracy,
                    "Classification_Loss": classification_loss,
                    "Robustness_Loss": robustness_loss,
                    "Concept_Loss": concept_loss,
                    "Loss": total_loss,
                    "Step": self.current_iter,
                }
            )

    def finalize(self):
        """Finalize all necessary operations before exiting training.

        Saves checkpoint.
        """
        logging.info("Please wait while we finalize...")
        self.save_checkpoint()

    def summarize(self):
        """Print summary of given model."""
        logging.info(self.model)
        train_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Trainable Parameters: {train_params}\n")


def handle_integer_input(input, desired_len):
    """
    Checks if the input is an integer or a list.
    If an integer, it is replicated the number of  desired times
    If a tuple, the tuple is returned as it is
    Parameters
    ----------
    input : int, tuple
        The input can be either a tuple of parameters or a single parameter to be replicated
    desired_len : int
        The length of the desired list
    Returns
    -------
    input : tuple[int]
        a tuple of parameters which has the proper length.
    """
    if type(input) is int:
        return (input,) * desired_len
    elif type(input) is tuple:
        if len(input) != desired_len:
            raise AssertionError(
                "The sizes of the parameters for the CNN conceptizer do not match."
                f"Expected '{desired_len}', but got '{len(input)}'"
            )
        else:
            return input
    else:
        raise TypeError(
            f"Wrong type of the parameters. Expected tuple or int but got '{type(input)}'"
        )


def get_dataloader(config):
    """Dispatcher that calls dataloader function depending on the configs.
    Parameters
    ----------
    config : SimpleNameSpace
        Contains configs values. Needs to at least have a `dataloader` field.

    Returns
    -------
    Corresponding dataloader.
    """
    return load_mnist(**config.__dict__)


def load_mnist(data_path, batch_size, num_workers=0, valid_size=0.1, **kwargs):
    """
    Load mnist data.
    Loads mnist dataset and performs the following preprocessing operations:
        - converting to tensor
        - standard mnist normalization so that values are in (0, 1)
    Parameters
    ----------
    data_path: str
        Location of mnist data.
    batch_size: int
        Batch size.
    num_workers: int
        the number of  workers to be used by the Pytorch DataLoaders
    valid_size : float
        a float between 0.0 and 1.0 for the percent of samples to be used for validation
    Returns
    -------
    train_loader
        Dataloader for training set.
    valid_loader
        Dataloader for validation set.
    test_loader
        Dataloader for testing set.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    train_set = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        data_path, train=False, download=True, transform=transform
    )

    train_size = len(train_set)
    split = int(np.floor(valid_size * train_size))
    indices = list(range(train_size))
    train_sampler = SubsetRandomSampler(indices[split:])
    valid_sampler = SubsetRandomSampler(indices[:split])

    dataloader_args = dict(
        batch_size=batch_size, num_workers=num_workers, drop_last=True
    )
    train_loader = DataLoader(train_set, sampler=train_sampler, **dataloader_args)
    valid_loader = DataLoader(train_set, sampler=valid_sampler, **dataloader_args)
    test_loader = DataLoader(test_set, shuffle=False, **dataloader_args)

    return train_loader, valid_loader, test_loader


def mse_l1_sparsity(x, x_hat, concepts, sparsity_reg):
    """Sum of Mean Squared Error and L1 norm weighted by sparsity regularization parameter
    Parameters
    ----------
    x : torch.tensor
        Input data to the encoder.
    x_hat : torch.tensor
        Reconstructed input by the decoder.
    concepts : torch.Tensor
        Concept (latent code) activations.
    sparsity_reg : float
        Regularizer (xi) for the sparsity term.
    Returns
    -------
    loss : torch.tensor
        Concept loss
    """
    return F.mse_loss(x_hat, x.detach()) + sparsity_reg * torch.abs(concepts).sum()


def mnist_robustness_loss(x, aggregates, concepts, relevances):
    """Computes Robustness Loss for MNIST data

    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design
    Parameters
    ----------
    x            : torch.tensor
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor
                 Aggregates from SENN as (batch_size x num_classes x concept_dim)
    concepts     : torch.tensor
                 Concepts from Conceptizer as (batch_size x num_concepts x concept_dim)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)

    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    # concept_dim is always 1
    concepts = concepts.squeeze(-1)
    aggregates = aggregates.squeeze(-1)

    batch_size = x.size(0)
    num_concepts = concepts.size(1)
    num_classes = aggregates.size(1)

    # Jacobian of aggregates wrt x
    jacobians = []
    for i in range(num_classes):
        grad_tensor = torch.zeros(batch_size, num_classes).to(x.device)
        grad_tensor[:, i] = 1.0
        j_yx = torch.autograd.grad(
            outputs=aggregates,
            inputs=x,
            grad_outputs=grad_tensor,
            create_graph=True,
            only_inputs=True,
        )[0]
        # bs x 1 x 28 x 28 -> bs x 784 x 1
        jacobians.append(j_yx.view(batch_size, -1).unsqueeze(-1))
    # bs x num_features x num_classes (bs x 784 x 10)
    J_yx = torch.cat(jacobians, dim=2)

    # Jacobian of concepts wrt x
    jacobians = []
    for i in range(num_concepts):
        grad_tensor = torch.zeros(batch_size, num_concepts).to(x.device)
        grad_tensor[:, i] = 1.0
        j_hx = torch.autograd.grad(
            outputs=concepts,
            inputs=x,
            grad_outputs=grad_tensor,
            create_graph=True,
            only_inputs=True,
        )[0]
        # bs x 1 x 28 x 28 -> bs x 784 x 1
        jacobians.append(j_hx.view(batch_size, -1).unsqueeze(-1))
    # bs x num_features x num_concepts
    J_hx = torch.cat(jacobians, dim=2)

    # bs x num_features x num_classes
    robustness_loss = J_yx - torch.bmm(J_hx, relevances)

    return robustness_loss.norm(p="fro")


def init_trainer(config_file, best_model=False):
    """Instantiate the Trainer class based on the config parameters
    Parameters
    ----------
    config_file: str
        filename of the json config with all experiment parameters
    best_model: bool
        whether to load the previously trained best model
    Returns
    -------
    trainer: SENN_Trainer
        Trainer for SENN or DiSENNTrainer for DiSENN
    """
    with open(config_file, "r") as f:
        config = json.load(f)

    if best_model:
        config["load_checkpoint"] = BEST_MODEL_FILENAME

    logging.info("==================================================")
    logging.info(f" EXPERIMENT: {config['exp_name']}")
    logging.info("==================================================")
    logging.info(config)
    config = SimpleNamespace(**config)
    # create the trainer class and init with config
    trainer = SENN_Trainer(config)
    return trainer
