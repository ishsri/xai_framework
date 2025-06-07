# xai_methods/grad_cam.py

import torch
from captum.attr import LayerGradCam

def apply_grad_cam(model, input_tensor, target_class, layer):
    gradcam = LayerGradCam(model, getattr(model, layer))
    attr = gradcam.attribute(input_tensor, target=target_class)
    return attr.squeeze().detach().numpy()
