import numpy as np
import matplotlib.cm as cm

from PIL import Image


def plot_heatmap_full_sized(tensor, filename):
    """
    Save a square mask tensor as an RGB image.

    Parameters:
    mask (torch.Tensor): A boolean tensor of shape (n, n) where True is black and False is white.
    filename (str): The name of the file to save the image as.
    """

    # Convert the tensor to a numpy array
    tensor_np = tensor.cpu().numpy()

    # Normalize the tensor to the range [0, 1] for applying colormap
    vmin, vmax = np.percentile(tensor_np, [0, 98])
    tensor_np = (tensor_np - vmin) / (vmax - vmin)
    tensor_np = tensor_np.clip(0, 1)  # Values should be in [0, 1]

    # Apply a color map (e.g., 'viridis') using matplotlib
    colormap = cm.get_cmap('viridis')  # Choose any other colormap you like (e.g., 'plasma', 'inferno')
    tensor_colored = colormap(tensor_np)  # This returns an RGBA array

    # Convert the RGBA values to an image (ignore the alpha channel)
    tensor_colored = (tensor_colored[:, :, :3] * 255).astype(np.uint8)

    # Create and save the color-mapped image using PIL
    image = Image.fromarray(tensor_colored)
    image.save(filename + ".png")


def plot_heatmap(tensor, filename, title=None):
    import matplotlib.pyplot as plt

    vmin, vmax = np.percentile(tensor.cpu().numpy(), [0, 98])

    m, n = tensor.shape

    plt.figure(figsize=(8, 8), dpi=100)
    plt.imshow(tensor.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, n, 0, m])
    plt.colorbar(label='Values')
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    plt.tight_layout(pad=0)

    plt.savefig(filename, format='pdf', dpi=100)
    plt.close()
