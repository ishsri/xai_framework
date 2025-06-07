# main.py

import torch
from xai_methods import utils, grad_cam, occlusion, integrated_gradients, lrp, smoothgradpp, guided_backprop
import numpy as np
import matplotlib.pyplot as plt

METHOD_MAP = {
    "grad_cam": grad_cam.apply_grad_cam,
    "occlusion": occlusion.apply_occlusion,
    "integrated_gradients": integrated_gradients.apply_ig,
    "lrp": lrp.apply_lrp,
    "smoothgradpp": smoothgradpp.apply_smoothgradpp,
    "guided_backprop": guided_backprop.apply_guided_backprop
}

def run_xai(image_path, model_path, arch, layer, target_class, methods):
    model = utils.load_model(arch, model_path)
    input_tensor = utils.preprocess_image(image_path)

    results = {}
    for method in methods:
        print(f"Running {method}...")
        if method == "grad_cam":
            result = METHOD_MAP[method](model, input_tensor, target_class, layer)
        else:
            result = METHOD_MAP[method](model, input_tensor, target_class)
        results[method] = result

    return results
