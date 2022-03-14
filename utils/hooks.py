from pathlib import Path
import numpy as np
import os


def create_forward_hook(name: str, save_dir: Path):
    """
    Registers a hook with a given name and save the associated tensors
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


def get_saved_representations(name: str, save_dir: Path) -> np.ndarray:
    with (save_dir/f"{name}.npy").open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        out = np.load(f)
        while f.tell() < fsz:
            out = np.vstack((out, np.load(f)))
    return out


