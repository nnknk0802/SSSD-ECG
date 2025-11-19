"""
SSSD-ECG Model Wrapper
Provides a clean interface for training and inference with the SSSD-ECG diffusion model.

Usage:
    # Training
    model = SSSDECG(config_path="config/config_SSSD_ECG.json")
    loss = model(x, y)

    # Inference
    pred = model.generate(labels=y, num_samples=10)
"""

import os
import json
import torch
import torch.nn as nn
from models.SSSD_ECG import SSSD_ECG as SSSD_ECG_Base
from utils.util import calc_diffusion_hyperparams, training_loss_label, sampling_label


class SSSDECG(nn.Module):
    """
    Wrapper class for SSSD-ECG diffusion model with clean interface.

    This class handles:
    - Loading configuration
    - Initializing diffusion hyperparameters
    - Training loss computation
    - Sample generation

    Args:
        config_path (str): Path to configuration JSON file. Default: "config/config_SSSD_ECG.json"
        device (str): Device to run the model on. Default: "cuda" if available, else "cpu"
        num_classes (int, optional): Number of label classes. If provided, overrides the config value.
                                     Use this when your data has a different number of classes than the config.
    """

    def __init__(self, config_path=None, device=None, num_classes=None):
        super(SSSDECG, self).__init__()

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load configuration
        if config_path is None:
            # Default config path relative to this file
            config_path = os.path.join(
                os.path.dirname(__file__),
                "config",
                "config_SSSD_ECG.json"
            )

        with open(config_path) as f:
            self.config = json.load(f)

        # Extract configs
        self.diffusion_config = self.config["diffusion_config"]
        self.model_config = self.config["wavenet_config"]

        # Override number of classes if provided
        if num_classes is not None:
            self.model_config["label_embed_classes"] = num_classes
            print(f"Using {num_classes} classes (overriding config value of {self.config['wavenet_config'].get('label_embed_classes', 'N/A')})")

        # Calculate diffusion hyperparameters
        self.diffusion_hyperparams = calc_diffusion_hyperparams(
            T=self.diffusion_config["T"],
            beta_0=self.diffusion_config["beta_0"],
            beta_T=self.diffusion_config["beta_T"]
        )

        # Move diffusion hyperparameters to device
        for key in self.diffusion_hyperparams:
            if key != "T":
                self.diffusion_hyperparams[key] = self.diffusion_hyperparams[key].to(self.device)

        # Initialize the base model
        self.model = SSSD_ECG_Base(**self.model_config).to(self.device)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # For tracking training mode
        self._is_training = True

    def forward(self, x, y):
        """
        Compute training loss for a batch of data.

        Args:
            x (torch.Tensor): Input ECG signals, shape=(batch_size, channels, length)
            y (torch.Tensor): Labels, shape=(batch_size, num_classes) for multi-label
                             or (batch_size,) for single-label (will be converted to one-hot)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)

        # Handle label format - convert to float if needed
        if y.dtype != torch.float32:
            y = y.float()

        # If y is 1D, convert to one-hot encoding
        if len(y.shape) == 1:
            num_classes = self.model_config.get("label_embed_classes", 71)
            y_onehot = torch.zeros(y.size(0), num_classes, device=self.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
            y = y_onehot

        # Compute loss
        X = (x, y)
        loss = training_loss_label(
            self.model,
            self.loss_fn,
            X,
            self.diffusion_hyperparams
        )

        return loss

    def generate(self, labels=None, num_samples=1, return_numpy=False):
        """
        Generate ECG samples using the diffusion model.

        Args:
            labels (torch.Tensor, optional): Conditioning labels
                - shape=(num_samples, num_classes) for multi-label
                - shape=(num_samples,) for class indices (will be converted to one-hot)
                - If None, random labels will be generated
            num_samples (int): Number of samples to generate. Default: 1
                              Only used if labels is None
            return_numpy (bool): If True, return numpy array instead of torch tensor

        Returns:
            torch.Tensor or np.ndarray: Generated ECG signals
                shape=(num_samples, channels, length)
        """
        # Ensure model is in eval mode
        was_training = self.model.training
        self.model.eval()

        # Handle labels
        if labels is None:
            # Generate random labels
            num_classes = self.model_config.get("label_embed_classes", 71)
            labels = torch.randint(0, num_classes, (num_samples,), device=self.device)
        else:
            labels = labels.to(self.device)
            num_samples = labels.size(0)

        # Convert labels to proper format if needed
        if labels.dtype != torch.float32:
            labels = labels.float()

        # If labels is 1D, convert to one-hot
        if len(labels.shape) == 1:
            num_classes = self.model_config.get("label_embed_classes", 71)
            labels_onehot = torch.zeros(num_samples, num_classes, device=self.device)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels = labels_onehot

        # Determine output size
        channels = self.model_config["out_channels"]
        length = self.config.get("trainset_config", {}).get("segment_length", 1000)
        size = (num_samples, channels, length)

        # Generate samples
        with torch.no_grad():
            samples = sampling_label(
                self.model,
                size,
                self.diffusion_hyperparams,
                cond=labels
            )

        # Restore training mode
        if was_training:
            self.model.train()

        # Convert to numpy if requested
        if return_numpy:
            return samples.cpu().numpy()

        return samples

    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint .pkl file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    def save_checkpoint(self, checkpoint_path, optimizer=None, epoch=None):
        """
        Save model weights to a checkpoint file.

        Args:
            checkpoint_path (str): Path to save the checkpoint
            optimizer (torch.optim.Optimizer, optional): Optimizer state to save
            epoch (int, optional): Current epoch/iteration number
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def get_model_size(self):
        """
        Get the number of parameters in the model.

        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_model_size(self):
        """Print model size information."""
        num_params = self.get_model_size()
        print(f"Model Parameters: {num_params / 1e6:.2f}M")
