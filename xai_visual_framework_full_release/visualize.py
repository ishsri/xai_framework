# visualize.py

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_composite(results, input_path, save_path="composite_output.png"):
    img = Image.open(input_path).resize((224, 224))
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(4 * (len(results) + 1), 4))
    
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (name, heatmap) in enumerate(results.items(), start=1):
        axes[i].imshow(img)
        axes[i].imshow(heatmap.sum(axis=0), cmap='jet', alpha=0.5)
        axes[i].set_title(name)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
