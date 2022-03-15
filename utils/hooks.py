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
    if (save_dir/f"{name}.npy").exists():  # Ensure that the hook is not going to append an existing file
        os.remove(save_dir/f"{name}.npy")
    def hook(module, input, output):
        save_arr = output.detach().cpu().numpy()  # Convert tensor to Numpy array
        save_arr = save_arr.reshape((save_arr.shape[0], -1))  # Flatten Numpy array
        with (save_dir/f"{name}.npy").open('ab') as f:
            np.save(f, save_arr)
    return hook


def register_hooks(model: nn.Module, save_dir: Path, prefix: str) -> tuple:
    """
    Registers all the hooks specified in the model
    Args:
        save_dir: directory where hooked arrays should be saved
        model: model for which hooks should be registered
        prefix: string identifier for the hook (useful for naming the file where activations are saved)

    Returns:
        a dictionary of the form {module_name : module}

    """
    module_dic = model.get_hooked_modules()
    handler_dic = {}
    for module_name in module_dic:
        hook_name = f"{prefix}_{module_name}"
        handler = module_dic[module_name].register_forward_hook(create_forward_hook(hook_name, save_dir))
        handler_dic[hook_name] = handler
    return module_dic, handler_dic


def remove_all_hooks(handler_dic: dict) -> None:
    """
    Removes the all the hooks in the handler_dic
    Args:
        handler_dic: dictionary of the form {hook_name : handler}
    Returns:

    """
    for handle in handler_dic.values():
        handle.remove()


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


