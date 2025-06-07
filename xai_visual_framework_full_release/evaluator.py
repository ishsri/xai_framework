# evaluator.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def deletion_score(input_tensor, attribution, model, target_class, steps=20):
    input_tensor = input_tensor.clone().detach()
    base = input_tensor.clone()
    mask = np.argsort(-attribution.flatten())  # descending order
    flattened = input_tensor.flatten()

    scores = []
    for i in range(1, steps + 1):
        n = int(i * len(mask) / steps)
        flattened[mask[:n]] = 0
        perturbed = flattened.view(input_tensor.shape)
        with torch.no_grad():
            output = model(perturbed)
            prob = torch.nn.functional.softmax(output, dim=1)[0, target_class].item()
        scores.append(prob)

    x = np.linspace(0, 1, steps)
    return x, scores, auc(x, scores)

def plot_deletion_curves(all_scores, save_path="deletion_curves.png"):
    plt.figure()
    for method, (x, y, score) in all_scores.items():
        plt.plot(x, y, label=f"{method} (AUC={score:.2f})")
    plt.xlabel("Fraction of removed important pixels")
    plt.ylabel("Model Confidence")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
