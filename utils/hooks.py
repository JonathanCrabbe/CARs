from pathlib import Path
import numpy as np
import os
import torch.nn as nn


def create_forward_hook(name: str, save_dir: Path):
    """
    Create a hook with a given name and save the associated tensors
    Args:
        name: name of the module
        save_dir: path where arrays are saved

    Returns:
        The hook to be registered
    """
    def hook(module, input, output):
        save_arr = output.detach().cpu().numpy()  # Convert tensor to Numpy array
        save_arr = save_arr.reshape((save_arr.shape[0], -1))  # Flatten Numpy array
        with (save_dir/f"{name}.npy").open('ab') as f:
            np.save(f, save_arr)
    return hook


def register_hooks(model: nn.Module, save_dir: Path) -> dict:
    """
    Registers all the hooks specified in the model
    Args:
        save_dir: directory where hooked arrays should be saved
        model: model for which hooks should be registered

    Returns:
        a dictionary of the form {module_name : module}

    """
    module_dic = model.get_hooked_modules()
    for module_name in module_dic:
        module_dic[module_name].register_forward_hook(create_forward_hook(module_name, save_dir))
    return module_dic


def get_saved_representations(name: str, save_dir: Path) -> np.ndarray:
    """
    Loads the representations saved via the hooks
    Args:
        name: name of the hooked module
        save_dir: directory where representations are stored

    Returns:
        the representations as an array
    """
    with (save_dir/f"{name}.npy").open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        out = np.load(f)
        while f.tell() < fsz:
            out = np.vstack((out, np.load(f)))
    return out


