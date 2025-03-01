import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_latent_codes(file_paths):
    """Loads latent codes from .npy files.
    
    Args:
        file_paths (list): List of file paths to the .npy files.
        
    Returns:
        list: A list of numpy arrays containing the latent codes.
    """
    latent_codes = []
    for path in file_paths:
        latent_codes.append(np.load(path))
    return latent_codes

def visualize_with_tsne(latent_codes_list, labels=None, perplexity=30.0):
    """Performs t-SNE dimensionality reduction and visualizes the latent codes.
    
    Args:
        latent_codes_list (list): A list of numpy arrays containing the latent codes.
        labels (list, optional): Labels for each group of latent codes.
        perplexity (float, optional): Perplexity parameter for t-SNE.
    """
    colors = ["#d06b7a", "#7afb1e", "#1b19d4"]  # New colors for the different groups "#2561b5"
    plt.figure(figsize=(10, 8))

    for i, latent_codes in enumerate(latent_codes_list):
        # Perform t-SNE
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300)
        tsne_results = tsne.fit_transform(latent_codes)

        # Plot the results
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors[i], label=f'Group {i+1}', alpha=0.8)

    plt.title('t-SNE visualization of latent codes')
    if labels is not None:
        plt.legend(title='Group labels')
    plt.show()

# Example usage:
file_paths = ['features_spikeVGG_passway_augmented.npy', 'features_spikeVGG_printer_augmented.npy', 'features_spikeVGG_hall_augmented.npy']
latent_codes_list = load_latent_codes(file_paths)
visualize_with_tsne(latent_codes_list)
